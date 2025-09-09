import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

class DQN(nn.Module):
    def __init__(self, input_size=200, hidden_size=256, output_size=3):
        """
        Deep Q-Network for Snake Game
        Input: State vector of size 200
        Output: Q-values for 3 actions (straight, turn right, turn left)
        """
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
    def __init__(self, state_size=200, action_size=3, lr=0.0001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=100000)
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = lr
        self.gamma = 0.98  # discount factor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Neural networks
        self.q_network = DQN(state_size, 256, action_size).to(self.device)
        self.target_network = DQN(state_size, 256, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Update target network
        self.update_target_network()

        # Training metrics
        self.losses = []

    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Choose action using epsilon-greedy policy"""
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        self.q_network.train()
        return np.argmax(q_values.cpu().data.numpy())

    def replay(self, batch_size=32):
        """Train the model on a batch of experiences"""
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

    def save(self, filepath):
        """Save model weights"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)

    def load(self, filepath):
        """Load model weights"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']

def train_agent(episodes=1000, batch_size=32, update_target_every=10):
    # Import the game (assuming it's in a file called snake_game.py)
    from snake_game_pixel import SnakeGameRL

    # Initialize game and agent
    game = SnakeGameRL(grid_size=10, display=False)
    agent = DQNAgent(state_size=game.get_state_size(), action_size=3)

    # Training metrics
    scores = deque(maxlen=100)
    max_score = 0

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
        max_score = max(max_score, game.score)

        # Update target network
        if episode % update_target_every == 0:
            agent.update_target_network()

        # Print progress
        if episode % 100 == 0:
            avg_score = np.mean(scores)
            print(f"Episode {episode}, Average Score: {avg_score:.2f}, "
                  f"Max Score: {max_score}, Epsilon: {agent.epsilon:.3f}")

        # Save model periodically
        if episode % 200 == 0 and episode > 0:
            agent.save(f'pixel/saved/snake_model_episode_pixel_{episode}.pth')

    return agent, scores

def test_agent(agent, num_games=10, display=True):
    from snake_game_pixel import SnakeGameRL

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

def plot_training_progress(scores, window=100):
    plt.figure(figsize=(12, 4))

    # Plot scores
    plt.subplot(1, 2, 1)
    plt.plot(scores)
    plt.title('Training Scores')
    plt.xlabel('Episode')
    plt.ylabel('Score')

    # Plot moving average
    plt.subplot(1, 2, 2)
    if len(scores) >= window:
        moving_avg = [np.mean(scores[i:i+window]) for i in range(len(scores)-window+1)]
        plt.plot(moving_avg)
        plt.title(f'Moving Average (window={window})')
        plt.xlabel('Episode')
        plt.ylabel('Average Score')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage
    print("Training DQN Agent for Snake Game...")
    print("=" * 50)

    # Train the agent
    agent, training_scores = train_agent(episodes=1000, batch_size=32)

    # Save the final model
    agent.save('pixel/saved/snake_model_pixel_final.pth')
    print("\nModel saved as 'snake_model_pixel_final.pth'")

    # Plot training progress
    plot_training_progress(list(training_scores))

    # Test the trained agent
    print("\nTesting trained agent...")
    print("=" * 50)
    test_scores = test_agent(agent, num_games=10, display=True)