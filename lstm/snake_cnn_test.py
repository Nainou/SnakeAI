import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from snake_game_objects import SnakeGameRL


# =========================
# Device
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# Helpers for state handling
# =========================
def maps_to_chw_numpy(maps_t: torch.Tensor) -> np.ndarray:
    """
    Accepts a torch tensor [C,H,W] of floats (0/1); returns numpy float32 [C,H,W].
    """
    if not isinstance(maps_t, torch.Tensor):
        raise TypeError("state_maps must be a torch.Tensor [C,H,W]")
    maps_t = maps_t.detach().to("cpu").float()
    assert maps_t.ndim == 3, f"state_maps must be [C,H,W], got {tuple(maps_t.shape)}"
    return maps_t.numpy().astype(np.float32)


def extras_to_numpy(extra_t: torch.Tensor) -> np.ndarray:
    """
    Accepts a torch tensor [9]; returns numpy float32 [9].

    In your SnakeGameRL, extra_information is:
      4 (dir one-hot) + 4 (tail dir one-hot) + 1 (hunger) = 9
    """
    if not isinstance(extra_t, torch.Tensor):
        raise TypeError("extra_information must be a torch.Tensor [9]")
    extra_t = extra_t.detach().to("cpu").float()
    assert extra_t.ndim == 1 and extra_t.shape[0] == 9, f"extra_information must be [9], got {tuple(extra_t.shape)}"
    return extra_t.numpy().astype(np.float32)


# =========================
# CNN + CoordConv + Dueling head
# =========================
class CNNDuelingQNet(nn.Module):
    """
    Non-recurrent CNN that works directly on your [C,H,W] maps + 9-dim extra vector.

    - Adds x,y CoordConv channels
    - 3 conv layers (no pooling) to keep full 10x10 resolution
    - MLP on extras
    - Concatenate features and feed to dueling MLP head
    """

    def __init__(self, c_in: int, h: int, w: int, action_size: int,
                 emb_maps: int = 128, emb_extra: int = 64, hidden: int = 256):
        super().__init__()
        self.c_in = c_in
        self.h = h
        self.w = w
        self.action_size = action_size

        # CoordConv: +2 channels (x,y)
        conv_in = c_in + 2

        self.conv = nn.Sequential(
            nn.Conv2d(conv_in, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.maps_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * h * w, emb_maps),
            nn.ReLU(),
        )

        self.extra_proj = nn.Sequential(
            nn.Linear(9, emb_extra),
            nn.ReLU(),
        )

        fused_dim = emb_maps + emb_extra

        # Dueling head
        self.adv = nn.Sequential(
            nn.Linear(fused_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_size),
        )
        self.val = nn.Sequential(
            nn.Linear(fused_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

        # Coord buffers (lazy-init per device/shape)
        self._xs = None
        self._ys = None

    def _coord_channels(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,C,H,W]
        return x_concat: [B,C+2,H,W] with normalized x,y channels in [-1,1].
        """
        B, C, H, W = x.shape
        device = x.device
        need_new = (
            self._xs is None or
            self._xs.device != device or
            self._xs.shape != (1, 1, H, W)
        )
        if need_new:
            ys = torch.linspace(-1, 1, steps=H, device=device).view(1, 1, H, 1).expand(1, 1, H, W)
            xs = torch.linspace(-1, 1, steps=W, device=device).view(1, 1, 1, W).expand(1, 1, H, W)
            self._xs = xs
            self._ys = ys
        xs = self._xs.expand(B, 1, H, W)
        ys = self._ys.expand(B, 1, H, W)
        return torch.cat([x, xs, ys], dim=1)

    def forward(self, maps_bchw: torch.Tensor, extras_b9: torch.Tensor) -> torch.Tensor:
        """
        maps_bchw: [B,C,H,W]
        extras_b9: [B,9]
        returns: Q-values [B, A]
        """
        x = self._coord_channels(maps_bchw)       # [B,C+2,H,W]
        x = self.conv(x)                          # [B,128,H,W]
        maps_feat = self.maps_proj(x)             # [B,emb_maps]

        extra_feat = self.extra_proj(extras_b9)   # [B,emb_extra]

        fused = torch.cat([maps_feat, extra_feat], dim=1)  # [B,emb_maps+emb_extra]

        A = self.adv(fused)                       # [B,A]
        V = self.val(fused)                       # [B,1]
        q = V + A - A.mean(dim=1, keepdim=True)   # dueling
        return q


# =========================
# Replay Buffer
# =========================
class ReplayBuffer:
    """
    Standard (non-recurrent) replay buffer.
    Stores tuples of:
      (maps[C,H,W], extras[9], action, reward, next_maps[C,H,W], next_extras[9], done)
    as numpy arrays / scalars.
    """

    def __init__(self, capacity: int = 100_000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, maps, extras, action, reward, next_maps, next_extras, done):
        self.buffer.append((maps, extras, action, reward, next_maps, next_extras, done))

    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        maps, extras, actions, rewards, next_maps, next_extras, dones = zip(*batch)
        return (
            np.stack(maps, axis=0),          # [B,C,H,W]
            np.stack(extras, axis=0),        # [B,9]
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_maps, axis=0),
            np.stack(next_extras, axis=0),
            np.array(dones, dtype=np.float32),
        )


# =========================
# DQN Agent (non-recurrent CNN)
# =========================
class CNNDQNAgent:
    def __init__(
        self,
        c_in: int,
        h: int,
        w: int,
        action_size: int,
        gamma: float = 0.98,
        lr: float = 1e-4,
        weight_decay: float = 1e-6,
        batch_size: int = 64,
        replay_capacity: int = 100_000,
        eps_start: float = 1.0,
        eps_end: float = 0.01,
        eps_decay: float = 0.9995,
        target_update_every: int = 50,
    ):
        self.c_in = c_in
        self.h = h
        self.w = w
        self.action_size = action_size

        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_every = target_update_every

        self.eps = eps_start
        self.eps_min = eps_end
        self.eps_decay = eps_decay

        self.q_net = CNNDuelingQNet(c_in, h, w, action_size).to(DEVICE)
        self.tgt_net = CNNDuelingQNet(c_in, h, w, action_size).to(DEVICE)
        self.tgt_net.load_state_dict(self.q_net.state_dict())
        self.tgt_net.eval()

        self.optim = optim.AdamW(self.q_net.parameters(), lr=lr, weight_decay=weight_decay)

        self.replay = ReplayBuffer(replay_capacity)

        self.train_steps = 0
        self.losses = []

    def act(self, obs_tuple) -> int:
        """
        obs_tuple = (state_maps [C,H,W] torch, extra_information [9] torch)
        epsilon-greedy on Q(s,·).
        """
        state_maps_t, extra_info_t = obs_tuple
        maps_np = maps_to_chw_numpy(state_maps_t)
        extras_np = extras_to_numpy(extra_info_t)

        if random.random() <= self.eps:
            return random.randrange(self.action_size)

        self.q_net.eval()
        with torch.no_grad():
            maps = torch.tensor(maps_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)   # [1,C,H,W]
            extras = torch.tensor(extras_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)  # [1,9]
            q = self.q_net(maps, extras)   # [1,A]
            action = int(q.argmax(dim=-1).item())
        self.q_net.train()
        return action

    def remember(self, maps, extras, action, reward, next_maps, next_extras, done):
        self.replay.push(maps, extras, action, reward, next_maps, next_extras, float(done))

    def soft_update_target(self, tau: float = 0.01):
        for tgt, src in zip(self.tgt_net.parameters(), self.q_net.parameters()):
            tgt.data.mul_(1.0 - tau).add_(tau * src.data)

    def train_step(self):
        if len(self.replay) < self.batch_size:
            return None

        (
            maps_np,
            extras_np,
            actions_np,
            rewards_np,
            next_maps_np,
            next_extras_np,
            dones_np,
        ) = self.replay.sample(self.batch_size)

        maps = torch.tensor(maps_np, dtype=torch.float32, device=DEVICE)           # [B,C,H,W]
        extras = torch.tensor(extras_np, dtype=torch.float32, device=DEVICE)       # [B,9]
        actions = torch.tensor(actions_np, dtype=torch.long, device=DEVICE)        # [B]
        rewards = torch.tensor(rewards_np, dtype=torch.float32, device=DEVICE)     # [B]
        next_maps = torch.tensor(next_maps_np, dtype=torch.float32, device=DEVICE) # [B,C,H,W]
        next_extras = torch.tensor(next_extras_np, dtype=torch.float32, device=DEVICE)
        dones = torch.tensor(dones_np, dtype=torch.float32, device=DEVICE)         # [B]

        # Q(s,a)
        q_all = self.q_net(maps, extras)                       # [B,A]
        q_taken = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)   # [B]

        # Double DQN targets
        with torch.no_grad():
            q_next_online = self.q_net(next_maps, next_extras)         # [B,A]
            a_star = q_next_online.argmax(dim=1, keepdim=True)         # [B,1]

            q_next_tgt = self.tgt_net(next_maps, next_extras)          # [B,A]
            next_q = q_next_tgt.gather(1, a_star).squeeze(1)           # [B]

            target = rewards + (1.0 - dones) * self.gamma * next_q

        loss = F.smooth_l1_loss(q_taken, target)

        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optim.step()

        self.losses.append(float(loss.item()))
        self.train_steps += 1

        # periodic target update (soft)
        if self.train_steps % self.target_update_every == 0:
            self.soft_update_target(tau=0.05)

        # epsilon decay
        if self.eps > self.eps_min:
            self.eps *= self.eps_decay
            if self.eps < self.eps_min:
                self.eps = self.eps_min

        return float(loss.item())


# =========================
# Training loop
# =========================
def train_cnn_agent(
    episodes: int = 800,
    action_size: int = 3,
    print_every: int = 10,
):
    game = SnakeGameRL(display=False)

    # Probe state sizes
    state_maps_t, extra_info_t = game.get_state()  # maps: [C,H,W], extras: [9]
    maps0 = maps_to_chw_numpy(state_maps_t)
    extras0 = extras_to_numpy(extra_info_t)
    c_in, h, w = maps0.shape

    agent = CNNDQNAgent(c_in, h, w, action_size)

    scores_window = deque(maxlen=print_every)
    best_score = 0

    for ep in range(episodes):
        game.reset()
        obs = game.get_state()
        done = False
        total_reward = 0.0

        while not done:
            a = agent.act(obs)
            next_obs, r, done, info = game.step(a)

            # convert and store transition
            sm_t, ex_t = obs
            nsm_t, nex_t = next_obs

            maps_np = maps_to_chw_numpy(sm_t)
            extras_np = extras_to_numpy(ex_t)
            next_maps_np = maps_to_chw_numpy(nsm_t)
            next_extras_np = extras_to_numpy(nex_t)

            agent.remember(maps_np, extras_np, a, r, next_maps_np, next_extras_np, done)

            # one training step per env step (you can tweak)
            agent.train_step()

            obs = next_obs
            total_reward += r

        scores_window.append(game.score)
        if game.score > best_score:
            best_score = game.score

        if (ep + 1) % print_every == 0:
            avg_score = np.mean(scores_window) if len(scores_window) > 0 else 0.0
            print(
                f"Ep {ep + 1} | "
                f"avgScore:{avg_score:.2f} bestScore:{best_score} "
                f"eps:{agent.eps:.3f} replay:{len(agent.replay)}"
            )
            best_score = 0

    return agent


# =========================
# Test loop (rendered)
# =========================
def test_cnn_agent(agent: CNNDQNAgent, num_games: int = 5, render_delay: int = 5):
    game = SnakeGameRL(display=True, render_delay=render_delay)
    agent.eps = 0.0

    scores = []
    for i in range(num_games):
        game.reset()
        obs = game.get_state()
        done = False
        while not done:
            a = agent.act(obs)
            obs, r, done, _ = game.step(a)

            # render and allow closing the window
            game.render()
            try:
                import pygame
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        game.close()
                        return scores
            except Exception:
                pass

        scores.append(game.score)
        print(f"Test game {i + 1}: score={game.score}")

    try:
        game.close()
    except Exception:
        pass

    if scores:
        print(f"Test avg:{np.mean(scores):.2f} max:{max(scores)} min:{min(scores)}")
    return scores


# =========================
# Main
# =========================
if __name__ == "__main__":
    print("Training non-recurrent CNN DQN on SnakeGameRL object-feature maps...")
    agent = train_cnn_agent(episodes=10000, action_size=3)
    print("Testing trained CNN agent...")
    test_cnn_agent(agent, num_games=5, render_delay=5)
