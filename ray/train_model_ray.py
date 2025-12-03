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
    def __init__(self, input_size=21, hidden_size=128, output_size=3):
        # Deep Q-Network for Snake Game
        # Input: State vector of size 21
        # Output: Q-values for 3 actions (straight, turn right, turn left)
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, output_size)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class DQNAgent:
    def __init__(self, state_size=21, action_size=3, lr=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=100000)
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.learning_rate = lr
        self.gamma = 0.95  # discount factor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Neural networks
        self.q_network = DQN(state_size, 128, action_size).to(self.device)
        self.target_network = DQN(state_size, 128, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Update target network
        self.update_target_network()

        # Training metrics
        self.losses = []

    def update_target_network(self):
        # Copy weights from main network to target network
        self.target_network.load_state_dict(self.q_network.state_dict())

    def remember(self, state, action, reward, next_state, done):
        # Store experience in replay buffer
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # Choose action using epsilon-greedy policy
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        self.q_network.train()
        return np.argmax(q_values.cpu().data.numpy())

    def replay(self, batch_size=32):
        # Train the model on a batch of experiences
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.FloatTensor([e[4] for e in batch]).to(self.device)

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1)
        self.optimizer.step()

        self.losses.append(loss.item())

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filepath, include_metadata=True):
        # Save model weights with metadata in filename
        # Args:
        #   filepath: Path to save the model (can be directory or full path)
        #   include_metadata: If True, include architecture metadata in filename
        # Extract architecture info
        input_size = self.state_size
        hidden_size = self.q_network.fc1.out_features
        output_size = self.action_size

        # If include_metadata, create filename with metadata
        if include_metadata:
            path_obj = Path(filepath)

            # Extract extra info from filename if present
            extra_info = ""
            if path_obj.suffix:  # Has extension, treat as file path
                if 'episode' in path_obj.stem:
                    ep_match = re.search(r'episode[_\s]*(\d+)', path_obj.stem)
                    if ep_match:
                        extra_info = f"ep{ep_match.group(1)}"
                elif 'final' in path_obj.stem.lower():
                    extra_info = "final"
                # Use parent directory for new filename
                save_dir = path_obj.parent
            else:
                # No extension, treat as directory
                save_dir = path_obj

            # Create new filename with metadata
            # Format: ray_{input_size}_{hidden_size}_{output_size}_{extra_info}.pth
            parts = ["ray", str(input_size), str(hidden_size), str(output_size)]
            if extra_info:
                parts.append(extra_info)
            new_filename = "_".join(parts) + ".pth"

            filepath = save_dir / new_filename

        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)

    def load(self, filepath):
        # Load model weights
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']

def train_agent(episodes=1000, batch_size=32, update_target_every=10):
    # Import the game (assuming it's in a file called snake_game.py)
    from snake_game_ray import SnakeGameRL

    # Initialize game and agent
    game = SnakeGameRL(grid_size=10, display=False)
    agent = DQNAgent(state_size=game.get_state_size(), action_size=3)

    # Training metrics
    scores = deque(maxlen=100)  # For moving average calculation
    all_scores = []  # Track all scores
    epoch_averages = []  # Track average scores per epoch (every 100 episodes)
    epoch_max_scores = []  # Track max scores per epoch (every 100 episodes)
    epoch_times = []  # Track time per epoch (every 100 episodes)
    max_score = 0
    epoch_max_score = 0  # Max score for current epoch

    # Timing metrics
    start_time = time.time()
    epoch_start_time = start_time

    for episode in range(episodes):
        state = game.reset()
        total_reward = 0

        while not game.done:
            # Choose action
            action = agent.act(state)

            # Take action
            next_state, reward, done, info = game.step(action)

            # Store experience
            agent.remember(state, action, reward, next_state, done)

            # Update state
            state = next_state
            total_reward += reward

            # Train model
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        # Store score
        scores.append(game.score)
        all_scores.append(game.score)
        max_score = max(max_score, game.score)
        epoch_max_score = max(epoch_max_score, game.score)

        # Update target network
        if episode % update_target_every == 0:
            agent.update_target_network()

        # Print progress and track epoch averages and max scores
        if episode % 100 == 0:
            avg_score = np.mean(scores)
            epoch_averages.append(avg_score)
            epoch_max_scores.append(epoch_max_score)

            # Calculate epoch time
            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time
            epoch_times.append(epoch_time)
            epoch_start_time = epoch_end_time  # Reset for next epoch

            print(f"Episode {episode}, Average Score: {avg_score:.2f}, "
                  f"Max Score: {max_score}, Epoch Max: {epoch_max_score}, Epsilon: {agent.epsilon:.3f}")
            epoch_max_score = 0  # Reset for next epoch

        # Save model periodically
        if episode % 500 == 0 and episode > 0:
            # Save with metadata in filename
            agent.save(f'ray/saved/snake_model_episode_{episode}.pth')

    # Record final epoch max score if training didn't end at a multiple of 100
    if episodes % 100 != 0:
        epoch_max_scores.append(epoch_max_score)
        # Record final epoch time
        final_epoch_time = time.time() - epoch_start_time
        epoch_times.append(final_epoch_time)

    # Calculate total training time
    total_time = time.time() - start_time

    # Calculate statistics
    avg_epoch_time = np.mean(epoch_times) if epoch_times else 0
    min_epoch_time = min(epoch_times) if epoch_times else 0
    max_epoch_time = max(epoch_times) if epoch_times else 0
    final_avg_score = np.mean(all_scores) if all_scores else 0

    training_stats = {
        'total_episodes': episodes,
        'total_time': total_time,
        'avg_epoch_time': avg_epoch_time,
        'min_epoch_time': min_epoch_time,
        'max_epoch_time': max_epoch_time,
        'epoch_times': epoch_times,
        'final_avg_score': final_avg_score,
        'final_max_score': max_score,
        'final_min_score': min(all_scores) if all_scores else 0,
        'epoch_averages': epoch_averages,
        'epoch_max_scores': epoch_max_scores
    }

    return agent, all_scores, epoch_averages, epoch_max_scores, training_stats

def test_agent(agent, num_games=10, display=True):
    from snake_game_ray import SnakeGameRL

    game = SnakeGameRL(grid_size=10, display=display, render_delay=5)
    agent.epsilon = 0  # No exploration during testing

    scores = []

    for i in range(num_games):
        state = game.reset()

        while not game.done:
            action = agent.act(state)
            state, reward, done, info = game.step(action)

            if display:
                game.render()

                # Check for quit event
                import pygame
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        game.close()
                        return scores

        scores.append(game.score)
        print(f"Game {i+1}: Score = {game.score}")

    if display:
        game.close()

    print(f"\nAverage Score: {np.mean(scores):.2f}")
    print(f"Max Score: {max(scores)}")
    print(f"Min Score: {min(scores)}")

    return scores

def plot_training_progress(all_scores, epoch_averages, window=100):
    """Plot the moving average scores across all episodes"""
    plt.figure(figsize=(10, 6))

    # Calculate moving average for all episodes using a rolling window
    if len(all_scores) >= window:
        moving_avg = [np.mean(all_scores[i:i+window]) for i in range(len(all_scores)-window+1)]
        episode_numbers = list(range(window-1, len(all_scores)))
        plt.plot(episode_numbers, moving_avg, linewidth=2, label=f'Moving Average (window={window})')
    else:
        # If not enough data, just plot what we have
        plt.plot(all_scores, linewidth=2, label='Scores')

    plt.title('Training Progress - Moving Average', fontsize=14, fontweight='bold')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Average Score', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_epoch_max_scores(epoch_max_scores):
    """Plot the max score per epoch"""
    plt.figure(figsize=(10, 6))
    epochs = [i * 100 for i in range(len(epoch_max_scores))]
    plt.plot(epochs, epoch_max_scores, marker='o', linewidth=2, markersize=6, color='red')
    plt.title('Max Score per Epoch', fontsize=14, fontweight='bold')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Max Score', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def print_training_statistics(stats):
    """Print comprehensive training statistics"""
    print("\n" + "=" * 60)
    print("TRAINING STATISTICS")
    print("=" * 60)

    # Episode information
    print(f"\nTotal Episodes: {stats['total_episodes']}")

    # Time statistics
    print(f"\n--- Time Statistics ---")
    print(f"Total Training Time: {stats['total_time']:.2f} seconds ({stats['total_time']/60:.2f} minutes)")
    print(f"Average Time per Episode: {stats['total_time']/stats['total_episodes']:.3f} seconds")

    if stats['epoch_times']:
        print(f"\nEpoch Time Statistics (per 100 episodes):")
        print(f"  Average Epoch Time: {stats['avg_epoch_time']:.2f} seconds ({stats['avg_epoch_time']/60:.2f} minutes)")
        print(f"  Min Epoch Time: {stats['min_epoch_time']:.2f} seconds ({stats['min_epoch_time']/60:.2f} minutes)")
        print(f"  Max Epoch Time: {stats['max_epoch_time']:.2f} seconds ({stats['max_epoch_time']/60:.2f} minutes)")

    # Score statistics
    print(f"\n--- Score Statistics ---")
    print(f"Final Average Score: {stats['final_avg_score']:.2f}")
    print(f"Final Max Score: {stats['final_max_score']}")
    print(f"Final Min Score: {stats['final_min_score']}")

    if stats['epoch_averages']:
        print(f"\nEpoch Average Scores:")
        print(f"  First Epoch Average: {stats['epoch_averages'][0]:.2f}")
        print(f"  Last Epoch Average: {stats['epoch_averages'][-1]:.2f}")
        print(f"  Best Epoch Average: {max(stats['epoch_averages']):.2f}")
        print(f"  Improvement: {stats['epoch_averages'][-1] - stats['epoch_averages'][0]:.2f}")

    if stats['epoch_max_scores']:
        print(f"\nEpoch Max Scores:")
        print(f"  First Epoch Max: {stats['epoch_max_scores'][0]}")
        print(f"  Last Epoch Max: {stats['epoch_max_scores'][-1]}")
        print(f"  Best Epoch Max: {max(stats['epoch_max_scores'])}")

    print("=" * 60)

if __name__ == "__main__":
    # Example usage
    print("Training DQN Agent for Snake Game...")
    print("=" * 50)

    # Train the agent
    agent, training_scores, epoch_averages, epoch_max_scores, training_stats = train_agent(episodes=2000, batch_size=64)

    # Save the final model (will be renamed with metadata)
    agent.save('ray/saved/snake_model_final.pth')
    # Get the actual saved filename
    from pathlib import Path
    saved_dir = Path('ray/saved')
    final_models = list(saved_dir.glob('ray_*_final.pth'))
    if final_models:
        print(f"\nModel saved as '{final_models[0].name}'")
    else:
        print("\nModel saved (check ray/saved/ directory)")

    # Print training statistics
    print_training_statistics(training_stats)

    # Plot training progress (moving average)
    plot_training_progress(training_scores, epoch_averages)

    # Plot max scores per epoch
    plot_epoch_max_scores(epoch_max_scores)