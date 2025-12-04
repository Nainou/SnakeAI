import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
from pathlib import Path
import re
import time


class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        Generic Deep Q-Network.

        - input_size: state dimension
        - output_size: number of actions
        """
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, output_size)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class DQNAgent:
    def __init__(self, state_size, action_size, lr=0.001):
        self.state_size = state_size
        self.action_size = action_size

        self.memory = deque(maxlen=100000)
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.9995
        self.gamma = 0.95
        self.learning_rate = lr

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_network = DQN(state_size, 128, action_size).to(self.device)
        self.target_network = DQN(state_size, 128, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        self.update_target_network()
        self.losses = []

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        self.q_network.train()
        return int(torch.argmax(q_values, dim=1).item())

    def replay(self, batch_size=64):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.FloatTensor([e[4] for e in batch]).to(self.device)

        # Q(s, a) for current states/actions
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        # max_a' Q_target(s', a')
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        loss = F.mse_loss(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        self.losses.append(loss.item())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filepath, include_metadata=True):
        input_size = self.state_size
        hidden_size = self.q_network.fc1.out_features
        output_size = self.action_size

        if include_metadata:
            path_obj = Path(filepath)
            extra_info = ""
            if path_obj.suffix:
                if "episode" in path_obj.stem:
                    ep_match = re.search(r"episode[_\s]*(\d+)", path_obj.stem)
                    if ep_match:
                        extra_info = f"ep{ep_match.group(1)}"
                elif "final" in path_obj.stem.lower():
                    extra_info = "final"
                save_dir = path_obj.parent
            else:
                save_dir = path_obj

            parts = ["frogger", str(input_size), str(hidden_size), str(output_size)]
            if extra_info:
                parts.append(extra_info)
            new_filename = "_".join(parts) + ".pth"
            filepath = save_dir / new_filename

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "q_network_state_dict": self.q_network.state_dict(),
                "target_network_state_dict": self.target_network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
            },
            filepath,
        )

    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint["epsilon"]


def train_agent(episodes=2000, batch_size=64, update_target_every=20):
    from frogger_game_ray import FroggerGameRL

    game = FroggerGameRL(grid_width=10, grid_height=12, display=False)
    state_size = game.get_state_size()
    action_size = 5  # stay, up, down, left, right

    agent = DQNAgent(state_size=state_size, action_size=action_size, lr=0.001)

    scores = deque(maxlen=10)
    all_scores = []
    epoch_averages = []
    epoch_max_scores = []
    epoch_times = []

    max_score = 0
    epoch_max_score = 0

    start_time = time.time()
    epoch_start_time = start_time

    for episode in range(episodes):
        state = game.reset()
        total_reward = 0.0

        while not game.done:
            action = agent.act(state)
            next_state, reward, done, info = game.step(action)

            agent.remember(state, action, reward, next_state, float(done))

            state = next_state
            total_reward += reward

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        scores.append(game.score)
        all_scores.append(game.score)
        max_score = max(max_score, game.score)
        epoch_max_score = max(epoch_max_score, game.score)

        if episode % update_target_every == 0:
            agent.update_target_network()

        if episode % 10 == 0:
            avg_score = np.mean(scores)
            epoch_averages.append(avg_score)
            epoch_max_scores.append(epoch_max_score)

            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time
            epoch_times.append(epoch_time)
            epoch_start_time = epoch_end_time

            print(
                f"Episode {episode}, Avg Score (last 10): {avg_score:.1f}, "
                f"Max Score: {max_score}, Epoch Max: {epoch_max_score}, "
                f"Epsilon: {agent.epsilon:.3f}"
            )
            epoch_max_score = 0

        if episode % 500 == 0 and episode > 0:
            agent.save("frogger/saved/frogger_model_episode_{episode}.pth")

    if episodes % 100 != 0:
        epoch_max_scores.append(epoch_max_score)
        final_epoch_time = time.time() - epoch_start_time
        epoch_times.append(final_epoch_time)

    total_time = time.time() - start_time

    training_stats = {
        "total_episodes": episodes,
        "total_time": total_time,
        "avg_epoch_time": np.mean(epoch_times) if epoch_times else 0.0,
        "min_epoch_time": min(epoch_times) if epoch_times else 0.0,
        "max_epoch_time": max(epoch_times) if epoch_times else 0.0,
        "epoch_times": epoch_times,
        "final_avg_score": np.mean(all_scores) if all_scores else 0.0,
        "final_max_score": max_score,
        "final_min_score": min(all_scores) if all_scores else 0.0,
        "epoch_averages": epoch_averages,
        "epoch_max_scores": epoch_max_scores,
    }

    return agent, all_scores, epoch_averages, epoch_max_scores, training_stats


def test_agent(agent, num_games=10, display=True):
    from frogger_game_ray import FroggerGameRL
    import pygame

    game = FroggerGameRL(grid_width=10, grid_height=12, display=display, render_delay=5)
    agent.epsilon = 0.0  # greedy policy

    scores = []

    for i in range(num_games):
        state = game.reset()

        while not game.done:
            action = agent.act(state)
            state, reward, done, info = game.step(action)

            if display:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        game.close()
                        return scores
                game.render()

        scores.append(game.score)
        print(f"Game {i + 1}: Score = {game.score}")

    if display:
        game.close()

    print(f"\nAverage Score: {np.mean(scores):.2f}")
    print(f"Max Score: {max(scores)}")
    print(f"Min Score: {min(scores)}")

    return scores


def plot_training_progress(all_scores, window=100):
    plt.figure(figsize=(10, 6))

    if len(all_scores) >= window:
        moving_avg = [
            np.mean(all_scores[i : i + window]) for i in range(len(all_scores) - window + 1)
        ]
        episode_numbers = list(range(window - 1, len(all_scores)))
        plt.plot(episode_numbers, moving_avg, linewidth=2, label=f"Moving Avg (window={window})")
    else:
        plt.plot(all_scores, linewidth=2, label="Scores")

    plt.title("Frogger Training Progress - Moving Average", fontsize=14, fontweight="bold")
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_epoch_max_scores(epoch_max_scores):
    plt.figure(figsize=(10, 6))
    epochs = [i * 10 for i in range(len(epoch_max_scores))]
    plt.plot(epochs, epoch_max_scores, marker="o", linewidth=2, markersize=6)
    plt.title("Frogger - Max Score per Epoch (10-episode chunks)", fontsize=14, fontweight="bold")
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Max Score", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def print_training_statistics(stats):
    print("\n" + "=" * 60)
    print("FROGGER TRAINING STATISTICS")
    print("=" * 60)
    print(f"\nTotal Episodes: {stats['total_episodes']}")

    print("\n--- Time ---")
    print(f"Total Time: {stats['total_time']:.2f} s ({stats['total_time'] / 60:.2f} min)")
    print(
        f"Avg Time/Episode: {stats['total_time'] / stats['total_episodes']:.3f} s"
    )

    if stats["epoch_times"]:
        print("\nEpoch Time (per 10 episodes):")
        print(
            f"  Avg Epoch Time: {stats['avg_epoch_time']:.2f} s "
            f"({stats['avg_epoch_time'] / 60:.2f} min)"
        )
        print(
            f"  Min Epoch Time: {stats['min_epoch_time']:.2f} s "
            f"({stats['min_epoch_time'] / 60:.2f} min)"
        )
        print(
            f"  Max Epoch Time: {stats['max_epoch_time']:.2f} s "
            f"({stats['max_epoch_time'] / 60:.2f} min)"
        )

    print("\n--- Scores ---")
    print(f"Final Avg Score: {stats['final_avg_score']:.2f}")
    print(f"Final Max Score: {stats['final_max_score']}")
    print(f"Final Min Score: {stats['final_min_score']}")

    if stats["epoch_averages"]:
        print("\nEpoch Avg Scores:")
        print(f"  First Epoch Avg: {stats['epoch_averages'][0]:.2f}")
        print(f"  Last Epoch Avg: {stats['epoch_averages'][-1]:.2f}")
        print(f"  Best Epoch Avg: {max(stats['epoch_averages']):.2f}")
        print(
            f"  Improvement: {stats['epoch_averages'][-1] - stats['epoch_averages'][0]:.2f}"
        )

    if stats["epoch_max_scores"]:
        print("\nEpoch Max Scores:")
        print(f"  First Epoch Max: {stats['epoch_max_scores'][0]}")
        print(f"  Last Epoch Max: {stats['epoch_max_scores'][-1]}")
        print(f"  Best Epoch Max: {max(stats['epoch_max_scores'])}")
    print("=" * 60)


if __name__ == "__main__":
    print("Training DQN Agent for Frogger Game...")
    print("=" * 50)

    agent, training_scores, epoch_averages, epoch_max_scores, training_stats = train_agent(
        episodes=2000, batch_size=64
    )

    agent.save("saved/frogger_model_final.pth")
    saved_dir = Path("saved")
    final_models = list(saved_dir.glob("frogger_*_final.pth"))
    if final_models:
        print(f"\nModel saved as '{final_models[0].name}'")
    else:
        print("\nModel saved (check saved/ directory)")

    print_training_statistics(training_stats)
    plot_training_progress(training_scores, window=100)
    plot_epoch_max_scores(epoch_max_scores)

    test_agent(agent, num_games=5, display=True)
