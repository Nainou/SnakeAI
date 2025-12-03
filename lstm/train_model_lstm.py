import math
import random
from collections import deque
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# =========================
# Config (edit if needed)
# =========================
GAME_MODULE = "snake_game_objects"   # file name without .py
GAME_CLASS_NAME = "SnakeGameRL"        # change this to your actual class name if different
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Replay / sequences
REPLAY_EPISODES = 800
SEQ_LEN = 16
BURN_IN = 4
BATCH_SIZE = 32

# Optimization
GAMMA = 0.98
LR = 1e-3
WEIGHT_DECAY = 1e-6
GRAD_CLIP = 1.0
HIDDEN_SIZE = 128
EMB_SIZE = 128
EMB_EXTRA = 128       # projection size for the 8-d extra_information
TAU = 0.05          # soft target update (Polyak)

# Exploration
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995

# Logging
PRINT_EVERY = 10
SAVE_PATH = "saved/snake_drqn_cnn_objects.pth"


# =========================
# Utils for obs handling
# =========================
def maps_to_chw_numpy(maps_t: torch.Tensor) -> np.ndarray:
    # Accepts a torch tensor [C,H,W] of floats (0/1); returns numpy float32 [C,H,W].
    # Accepts a torch tensor [C,H,W] of floats (0/1); returns numpy float32 [C,H,W].
    if not isinstance(maps_t, torch.Tensor):
        raise TypeError("state_maps must be a torch.Tensor [C,H,W]")
    maps_t = maps_t.detach().to("cpu").float()
    assert maps_t.ndim == 3, f"state_maps must be [C,H,W], got {tuple(maps_t.shape)}"
    return maps_t.numpy().astype(np.float32)


def extras_to_numpy(extra_t: torch.Tensor) -> np.ndarray:
    # Accepts a torch tensor [8]; returns numpy float32 [8].
    # Accepts a torch tensor [8]; returns numpy float32 [8].
    if not isinstance(extra_t, torch.Tensor):
        raise TypeError("extra_information must be a torch.Tensor [9]")
    extra_t = extra_t.detach().to("cpu").float()
    assert extra_t.ndim == 1 and extra_t.shape[0] == 9, f"extra_information must be [9], got {tuple(extra_t.shape)}"
    return extra_t.numpy().astype(np.float32)


# =========================
# Networks
# =========================
class CNNCoordEncoder(nn.Module):
    # CNN with CoordConv on top of 5-channel binary maps. We add 2 coordinate channels (x,y),
    # keep stride=1 and no pooling to retain the 10x10 resolution, then flatten and project.
    # Returns embedding per frame; extra_information is projected and concatenated outside.
    # CNN with CoordConv on top of 5-channel binary maps. We add 2 coordinate channels (x,y),
    # keep stride=1 and no pooling to retain the 10x10 resolution, then flatten and project.
    # Returns embedding per frame; extra_information is projected and concatenated outside.
    def __init__(self, in_channels: int, grid_h: int, grid_w: int, emb_size: int = EMB_SIZE):
        super().__init__()
        self.h = grid_h
        self.w = grid_w
        self.in_channels = in_channels + 2  # +2 for coord channels

        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * grid_h * grid_w, emb_size),
            nn.ReLU(),
        )

        # lazy buffers for coords (created on the fly to match device/dtype)
        self._xs = None
        self._ys = None

    def _coord_channels(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Build normalized coord channels in [-1,1]
        # x shape: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        device = x.device
        if (self._xs is None) or (self._xs.device != device) or (self._xs.shape != (1, 1, 1, H, W)):
            ys = torch.linspace(-1, 1, steps=H, device=device).view(1, 1, 1, H, 1).expand(1, 1, 1, H, W)
            xs = torch.linspace(-1, 1, steps=W, device=device).view(1, 1, 1, 1, W).expand(1, 1, 1, H, W)
            self._xs = xs
            self._ys = ys
        return self._xs.expand(B, T, 1, H, W), self._ys.expand(B, T, 1, H, W)

    def forward(self, maps_btchw: torch.Tensor) -> torch.Tensor:
        # maps_btchw: [B, T, C, H, W] float
        # returns: [B*T, emb_size]
        # maps_btchw: [B, T, C, H, W] float
        # returns: [B*T, emb_size]
        B, T, C, H, W = maps_btchw.shape
        xs, ys = self._coord_channels(maps_btchw)
        x = torch.cat([maps_btchw, xs, ys], dim=2)  # [B,T,C+2,H,W]
        x = x.view(B * T, C + 2, H, W)              # fold time into batch
        y = self.conv(x)                            # [B*T,64,H,W]
        z = self.proj(y)                            # [B*T,emb]
        return z


class DRQNCore(nn.Module):
    # CNNCoordEncoder(maps) + MLP(extras) -> concat -> LSTM -> Dueling Q-head.
    # Input per step: (maps: [C,H,W], extras: [8])
    # Batched over time: maps [B,T,C,H,W], extras [B,T,8]
    # CNNCoordEncoder(maps) + MLP(extras) -> concat -> LSTM -> Dueling Q-head.
    # Input per step: (maps: [C,H,W], extras: [8])
    # Batched over time: maps [B,T,C,H,W], extras [B,T,8]
    def __init__(self, c_in: int, grid_h: int, grid_w: int, action_size: int,
                 hidden_size: int = HIDDEN_SIZE, emb_size: int = EMB_SIZE, emb_extra: int = EMB_EXTRA):
        super().__init__()
        self.action_size = action_size
        self.hidden_size = hidden_size

        self.cnn = CNNCoordEncoder(c_in, grid_h, grid_w, emb_size=emb_size)
        self.extra_mlp = nn.Sequential(
            nn.Linear(9, emb_extra),
            nn.ReLU(),
        )

        self.lstm_in = emb_size + emb_extra
        self.lstm = nn.LSTM(
            input_size=self.lstm_in,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        # dueling head
        self.adv = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size)
        )
        self.val = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, maps_btchw: torch.Tensor, extras_bt8: torch.Tensor, hidden=None):
        # maps_btchw: [B,T,C,H,W]; extras_bt8: [B,T,8]
        # returns: q [B,T,A], hidden
        # maps_btchw: [B,T,C,H,W]; extras_bt8: [B,T,8]
        # returns: q [B,T,A], hidden
        B, T, C, H, W = maps_btchw.shape
        z_maps = self.cnn(maps_btchw)                  # [B*T, emb]
        z_maps = z_maps.view(B, T, -1)                 # [B,T,emb]

        z_extra = self.extra_mlp(extras_bt8)           # [B,T,emb_extra]

        z = torch.cat([z_maps, z_extra], dim=2)        # [B,T,emb+emb_extra]

        y, hidden = self.lstm(z, hidden)               # [B,T,H]
        A = self.adv(y)                                # [B,T,A]
        V = self.val(y)                                # [B,T,1]
        q = V + A - A.mean(dim=2, keepdim=True)        # [B,T,A]
        return q, hidden


# =========================
# Replay (episode-wise)
# =========================
class EpisodeBuffer:
    # Stores complete episodes. Each item:
    #   (maps[C,H,W], extras[8], action, reward, next_maps[C,H,W], next_extras[8], done)
    # All arrays are numpy float32 (maps: [C,H,W], extras: [8])
    # Stores complete episodes. Each item:
    #   (maps[C,H,W], extras[8], action, reward, next_maps[C,H,W], next_extras[8], done)
    # All arrays are numpy float32 (maps: [C,H,W], extras: [8])
    def __init__(self, capacity_episodes=REPLAY_EPISODES):
        self.capacity = capacity_episodes
        self.episodes: deque = deque(maxlen=capacity_episodes)
        self.current: List = []

    def start_episode(self):
        if self.current:
            self.end_episode()
        self.current = []

    def push(self, maps, extras, action, reward, next_maps, next_extras, done):
        self.current.append((maps, extras, action, reward, next_maps, next_extras, done))

    def end_episode(self):
        if self.current:
            self.episodes.append(self.current)
            self.current = []

    def num_episodes(self):
        return len(self.episodes)

    def __len__(self):
        return sum(len(ep) for ep in self.episodes)

    def sample_sequences(self, batch_size: int, seq_len: int, burn_in: int,
                         c_in: int, h: int, w: int) -> Optional[Tuple[torch.Tensor, ...]]:
        if len(self.episodes) == 0:
            return None

        batch = []
        for _ in range(batch_size):
            ep = random.choice(self.episodes)
            if len(ep) >= seq_len:
                start = random.randint(0, len(ep) - seq_len)
                seq = ep[start:start + seq_len]
            else:
                seq = ep[:]  # will pad
            batch.append(seq)

        # Prepare arrays
        maps_arr = np.zeros((batch_size, seq_len, c_in, h, w), dtype=np.float32)
        nxt_maps_arr = np.zeros((batch_size, seq_len, c_in, h, w), dtype=np.float32)
        extras_arr = np.zeros((batch_size, seq_len, 9), dtype=np.float32)
        nxt_extras_arr = np.zeros((batch_size, seq_len, 9), dtype=np.float32)
        actions_arr = np.zeros((batch_size, seq_len), dtype=np.int64)
        rewards_arr = np.zeros((batch_size, seq_len), dtype=np.float32)
        dones_arr = np.zeros((batch_size, seq_len), dtype=np.float32)
        masks_arr = np.zeros((batch_size, seq_len), dtype=np.float32)

        for i, seq in enumerate(batch):
            T = len(seq)
            if T >= seq_len:
                used = seq[:seq_len]
                mask = np.ones(seq_len, dtype=np.float32)
            else:
                used = seq + [(
                    np.zeros((c_in, h, w), dtype=np.float32),
                    np.zeros((9,), dtype=np.float32),
                    0, 0.0,
                    np.zeros((c_in, h, w), dtype=np.float32),
                    np.zeros((9,), dtype=np.float32),
                    1.0
                )] * (seq_len - T)
                mask = np.array([1.0] * T + [0.0] * (seq_len - T), dtype=np.float32)

            m, e, a, r, nm, ne, d = list(zip(*used))
            maps_arr[i] = np.stack(m, axis=0)
            nxt_maps_arr[i] = np.stack(nm, axis=0)
            extras_arr[i] = np.stack(e, axis=0)
            nxt_extras_arr[i] = np.stack(ne, axis=0)
            actions_arr[i] = np.array(a, dtype=np.int64)
            rewards_arr[i] = np.array(r, dtype=np.float32)
            dones_arr[i] = np.array(d, dtype=np.float32)
            masks_arr[i] = mask

        # torchify
        maps_t = torch.tensor(maps_arr, dtype=torch.float32, device=DEVICE)
        nxt_maps_t = torch.tensor(nxt_maps_arr, dtype=torch.float32, device=DEVICE)
        extras_t = torch.tensor(extras_arr, dtype=torch.float32, device=DEVICE)
        nxt_extras_t = torch.tensor(nxt_extras_arr, dtype=torch.float32, device=DEVICE)
        actions_t = torch.tensor(actions_arr, dtype=torch.long, device=DEVICE)
        rewards_t = torch.tensor(rewards_arr, dtype=torch.float32, device=DEVICE)
        dones_t = torch.tensor(dones_arr, dtype=torch.float32, device=DEVICE)
        masks_t = torch.tensor(masks_arr, dtype=torch.float32, device=DEVICE)

        return maps_t, extras_t, actions_t, rewards_t, nxt_maps_t, nxt_extras_t, dones_t, masks_t


# =========================
# Agent
# =========================
class DRQNAgent:
    def __init__(self, c_in: int, grid_h: int, grid_w: int, action_size: int):
        self.c_in = c_in
        self.h = grid_h
        self.w = grid_w
        self.action_size = action_size

        self.q_net = DRQNCore(c_in, grid_h, grid_w, action_size).to(DEVICE)
        self.tgt_net = DRQNCore(c_in, grid_h, grid_w, action_size).to(DEVICE)
        self.tgt_net.load_state_dict(self.q_net.state_dict())

        self.optim = optim.AdamW(self.q_net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        self.replay = EpisodeBuffer(REPLAY_EPISODES)

        self.gamma = GAMMA
        self.eps = EPS_START
        self.eps_min = EPS_END
        self.eps_decay = EPS_DECAY

        self.hidden = None
        self.losses = []

    # ---- recurrent helpers ----
    def reset_hidden(self, batch_size=1):
        h = torch.zeros(1, batch_size, HIDDEN_SIZE, device=DEVICE)
        c = torch.zeros(1, batch_size, HIDDEN_SIZE, device=DEVICE)
        self.hidden = (h, c)

    def _forward_step(self, maps_chw: np.ndarray, extras_9: np.ndarray) -> torch.Tensor:
        """
        Single-step forward for acting with recurrent state.
        maps_chw: numpy [C,H,W], extras_9: numpy [9]
        """
        m = torch.tensor(maps_chw, dtype=torch.float32, device=DEVICE).unsqueeze(0).unsqueeze(0)  # [1,1,C,H,W]
        e = torch.tensor(extras_9, dtype=torch.float32, device=DEVICE).view(1, 1, 9)              # [1,1,9]
        q_seq, self.hidden = self.q_net(m, e, self.hidden)  # [1,1,A]
        return q_seq.squeeze(0).squeeze(0)

    def act(self, obs_tuple) -> int:
        # obs_tuple = (state_maps [C,H,W] torch, extra_information [8] torch)
        state_maps_t, extra_info_t = obs_tuple
        maps_chw = maps_to_chw_numpy(state_maps_t)
        extras_9 = extras_to_numpy(extra_info_t)

        if random.random() <= self.eps:
            return random.randrange(self.action_size)
        q = self._forward_step(maps_chw, extras_9)
        return int(q.argmax(dim=-1).item())

    def remember(self, maps, extras, action, reward, next_maps, next_extras, done):
        self.replay.push(maps, extras, action, reward, next_maps, next_extras, float(done))

    @torch.no_grad()
    def soft_update_target(self, tau: float = TAU):
        for tgt, src in zip(self.tgt_net.parameters(), self.q_net.parameters()):
            tgt.data.mul_(1 - tau).add_(tau * src.data)

    def train_step(self) -> Optional[float]:
        sample = self.replay.sample_sequences(BATCH_SIZE, SEQ_LEN, BURN_IN, self.c_in, self.h, self.w)
        if sample is None:
            return None
        maps, extras, actions, rewards, nxt_maps, nxt_extras, dones, masks = sample  # shapes per their names
        B, T = actions.shape

        # zero initial hidden for each sampled sequence (truncated BPTT)
        h0 = torch.zeros(1, B, HIDDEN_SIZE, device=DEVICE)
        c0 = torch.zeros(1, B, HIDDEN_SIZE, device=DEVICE)

        # Q(s,a)
        q_seq, _ = self.q_net(maps, extras, (h0, c0))             # [B,T,A]
        q_taken = q_seq.gather(2, actions.unsqueeze(-1)).squeeze(-1)  # [B,T]

        # Double DQN target
        q_next_online, _ = self.q_net(nxt_maps, nxt_extras, (h0, c0))
        a_star = q_next_online.argmax(dim=2, keepdim=True)             # [B,T,1]
        with torch.no_grad():
            q_next_tgt, _ = self.tgt_net(nxt_maps, nxt_extras, (h0, c0))
            next_q = q_next_tgt.gather(2, a_star).squeeze(-1)          # [B,T]
            target = rewards + (1.0 - dones) * self.gamma * next_q

        # Burn-in mask
        if BURN_IN > 0:
            masks[:, :BURN_IN] = 0.0

        valid = masks > 0.0
        if valid.sum() == 0:
            return None

        loss = F.smooth_l1_loss(q_taken[valid], target[valid])

        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), GRAD_CLIP)
        self.optim.step()

        self.losses.append(float(loss.item()))

        # target soft update
        self.soft_update_target(TAU)

        return float(loss.item())


# =========================
# Training / Testing
# =========================
def _import_game():
    mod = __import__(GAME_MODULE, fromlist=[GAME_CLASS_NAME])
    cls = getattr(mod, GAME_CLASS_NAME)
    return cls


def train_agent(episodes=2000, action_size: int = 3):
    GameClass = _import_game()
    game = GameClass(display=False)

    # Peek first state for sizes
    state_maps_t, extra_info_t = game.get_state()  # per your interface
    maps0 = maps_to_chw_numpy(state_maps_t)        # [C,H,W]
    extras0 = extras_to_numpy(extra_info_t)        # [8]
    c_in, h, w = maps0.shape

    agent = DRQNAgent(c_in, h, w, action_size)
    scores = deque(maxlen=PRINT_EVERY)
    best = 0

    for ep in range(episodes):
        game.reset()
        obs = game.get_state()           # tuple (maps_torch[5,H,W], extras_torch[8])
        agent.replay.start_episode()
        agent.reset_hidden(batch_size=1)
        total_reward = 0.0

        done = False
        while not done:
            a = agent.act(obs)
            next_obs, r, done, info = game.step(a)

            # Convert & store
            sm_t, ex_t = obs
            nsm_t, nex_t = next_obs
            maps_np = maps_to_chw_numpy(sm_t)
            extras_np = extras_to_numpy(ex_t)
            next_maps_np = maps_to_chw_numpy(nsm_t)
            next_extras_np = extras_to_numpy(nex_t)

            agent.remember(maps_np, extras_np, a, r, next_maps_np, next_extras_np, done)

            obs = next_obs
            total_reward += r

            if agent.replay.num_episodes() >= 50:
                agent.train_step()

        agent.replay.end_episode()
        scores.append(game.score)
        best = max(best, game.score)

        # eps decay
        if agent.eps > agent.eps_min:
            agent.eps *= agent.eps_decay

        if ep % PRINT_EVERY == 0:
            avg = np.mean(scores) if len(scores) else 0.0
            print(f"Ep {ep} | avgScore:{avg:.1f} best:{best} eps:{agent.eps:.3f} "
                  f"replay_eps:{agent.replay.num_episodes()}")
            best = 0

    # Save final weights
    torch.save({
        "q": agent.q_net.state_dict(),
        "tgt": agent.tgt_net.state_dict(),
        "opt": agent.optim.state_dict(),
        "eps": agent.eps,
        "c_in": c_in, "h": h, "w": w,
        "action_size": action_size,
        "arch": {"hidden_size": HIDDEN_SIZE, "emb": EMB_SIZE, "emb_extra": EMB_EXTRA}
    }, SAVE_PATH)

    print(f"\nSaved model to {SAVE_PATH}")
    return agent, scores


def test_agent(agent, num_games=5, render_delay=5):
    GameClass = _import_game()
    game = GameClass(display=True, render_delay=render_delay)
    agent.eps = 0.0

    scores = []
    for i in range(num_games):
        game.reset()
        obs = game.get_state()
        agent.reset_hidden(batch_size=1)
        done = False
        while not done:
            a = agent.act(obs)
            obs, r, done, _ = game.step(a)
            try:
                game.render()
                import pygame
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        game.close()
                        return scores
            except Exception:
                pass
        scores.append(game.score)
        print(f"Test game {i+1}: score={game.score}")

    try:
        game.close()
    except Exception:
        pass

    print(f"Avg: {np.mean(scores):.2f} | Max: {max(scores)} | Min: {min(scores)}")
    return scores


if __name__ == "__main__":
    print("Training DRQN (CNN+CoordConv+LSTM, Double+Dueling, Burn-in) on object-feature maps...")
    agent, scores = train_agent(episodes=2000, action_size=3)
    print("Testing...")
    test_agent(agent, num_games=5, render_delay=5)
