import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import pygame

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

    def forward(self, x, return_activations=False):
        x = F.relu(self.fc1(x))
        fc1_activations = x.clone()

        x = F.relu(self.fc2(x))
        fc2_activations = x.clone()

        x = self.dropout(x)

        x = F.relu(self.fc3(x))
        fc3_activations = x.clone()

        x = self.fc4(x)
        output_values = x.clone()

        if return_activations:
            return x, {
                'fc1': fc1_activations,
                'fc2': fc2_activations,
                'fc3': fc3_activations,
                'output': output_values
            }
        return x

class DQNAgent:
    def __init__(self, state_size=21, action_size=3, lr=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=100000)
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
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

    def act(self, state, return_activations=False):
        # Choose action using epsilon-greedy policy
        if random.random() <= self.epsilon:
            action = random.randrange(self.action_size)
            if return_activations:
                # Return dummy activations for random actions
                dummy_activations = {
                    'fc1': torch.zeros(128),
                    'fc2': torch.zeros(128),
                    'output': torch.zeros(3)
                }
                return action, dummy_activations
            return action

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.q_network.eval()
        with torch.no_grad():
            if return_activations:
                q_values, activations = self.q_network(state_tensor, return_activations=True)
                # Convert activations to CPU and squeeze batch dimension
                activations = {k: v.cpu().squeeze(0) for k, v in activations.items()}
            else:
                q_values = self.q_network(state_tensor)
                activations = None
        self.q_network.train()

        action = np.argmax(q_values.cpu().data.numpy())

        if return_activations:
            return action, activations
        return action

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

    def save(self, filepath):
        # Save model weights
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

class NeuralNetworkVisualizer:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.colors = {
            'background': (20, 20, 30),
            'neuron_inactive': (50, 50, 70),
            'neuron_active': (255, 100, 100),
            'neuron_very_active': (255, 200, 100),
            'connection': (80, 80, 100),
            'text': (255, 255, 255),
            'layer_bg': (30, 30, 45)
        }
        self.font_size = 16
        self.small_font_size = 12

    def draw_network(self, surface, activations, state_values, action=None):
        # Draw the neural network with activations
        network_rect = pygame.Rect(0, 0, self.width, self.height)
        pygame.draw.rect(surface, self.colors['background'], network_rect)

        # Layer definitions - match the actual network architecture
        layers = [
            {'name': 'Input', 'size': 21, 'activations': state_values},
            {'name': 'Hidden 1', 'size': 128, 'activations': activations.get('fc1', torch.zeros(128))},
            {'name': 'Hidden 2', 'size': 128, 'activations': activations.get('fc2', torch.zeros(128))},
            {'name': 'Output', 'size': 3, 'activations': activations.get('output', torch.zeros(3))}
        ]

        # Calculate layout
        layer_width = self.width // (len(layers) + 1)
        margin_x = 20
        margin_y = 40

        # First pass: Calculate neuron positions
        neuron_positions = []
        for i, layer in enumerate(layers):
            positions = []
            x = i * layer_width + margin_x
            layer_height = self.height - 2 * margin_y
            neuron_spacing = layer_height / max(layer['size'], 1)

            for j in range(layer['size']):
                neuron_y = margin_y + j * neuron_spacing + neuron_spacing // 2
                neuron_x = x + layer_width // 2 - margin_x
                positions.append((neuron_x, neuron_y))
            neuron_positions.append(positions)

        # Draw layers and neurons
        for i, layer in enumerate(layers):
            x = i * layer_width + margin_x
            layer_height = self.height - 2 * margin_y

            # Draw layer background
            layer_rect = pygame.Rect(x - 10, margin_y - 10, layer_width - 20, layer_height + 20)
            pygame.draw.rect(surface, self.colors['layer_bg'], layer_rect, border_radius=5)

            # Draw layer title
            font = pygame.font.Font(None, self.font_size)
            text = font.render(layer['name'], True, self.colors['text'])
            text_rect = text.get_rect(centerx=x + layer_width // 2 - margin_x, y=margin_y - 30)
            surface.blit(text, text_rect)

            # Define state labels for input layer
            if layer['name'] == 'Input':
                state_labels = [
                    "Food Dir X", "Food Dir Y", "Dir UP", "Dir RIGHT", "Dir DOWN", "Dir LEFT",
                    "Danger UP", "Danger UR", "Danger R", "Danger DR",
                    "Danger DOWN", "Danger DL", "Danger L", "Danger UL",
                    "Dist X", "Dist Y", "Length"
                ]
            else:
                state_labels = None

            # Draw neurons
            for j, (neuron_x, neuron_y) in enumerate(neuron_positions[i]):
                if state_labels and j < len(state_labels):
                    label_text = state_labels[j]
                    small_font = pygame.font.Font(None, self.small_font_size)
                    value_text = small_font.render(label_text, True, self.colors['text'])
                    value_rect = value_text.get_rect(right=neuron_x - 15, centery=neuron_y)
                    surface.blit(value_text, value_rect)

                # Get activation value
                if layer['activations'] is not None and j < len(layer['activations']):
                    if isinstance(layer['activations'], torch.Tensor):
                        activation = float(layer['activations'][j])
                    else:
                        activation = float(layer['activations'][j])
                else:
                    activation = 0.0

                # Normalize activation
                if layer['name'] == 'Output':
                    if action is not None and j == action:
                        normalized_activation = 1.0
                    else:
                        normalized_activation = 0.0
                else:
                    if layer['activations'] is not None and len(layer['activations']) > 0:
                        activations_tensor = layer['activations']
                        if not isinstance(activations_tensor, torch.Tensor):
                            activations_tensor = torch.tensor(activations_tensor, dtype=torch.float32)
                        max_val = float(torch.max(torch.abs(activations_tensor)))
                        if max_val > 0:
                            normalized_activation = min(abs(activation) / max_val, 1.0)
                        else:
                            normalized_activation = 0.0
                    else:
                        normalized_activation = 0.0

                # Choose color based on activation
                if normalized_activation > 0.7:
                    color = self.colors['neuron_very_active']
                elif normalized_activation > 0.3:
                    color = self.colors['neuron_active']
                else:
                    inactive = np.array(self.colors['neuron_inactive'])
                    active = np.array(self.colors['neuron_active'])
                    color = tuple((inactive + (active - inactive) * normalized_activation).astype(int))

                # Draw neuron
                neuron_radius = 4 if layer['size'] > 100 else 6
                pygame.draw.circle(surface, color, (int(neuron_x), int(neuron_y)), neuron_radius)
                pygame.draw.circle(surface, self.colors['text'], (int(neuron_x), int(neuron_y)), neuron_radius, 1)

                # Draw activation value for output layer
                if layer['name'] == 'Output':
                    small_font = pygame.font.Font(None, self.small_font_size)
                    value_text = small_font.render(f'{activation:.2f}', True, self.colors['text'])
                    value_rect = value_text.get_rect(centerx=neuron_x + 25, centery=neuron_y)
                    surface.blit(value_text, value_rect)

        # Draw connections between layers
        for i in range(len(layers) - 1):
            current_layer = layers[i]
            next_layer = layers[i + 1]

            # Get activations for current layer
            current_activations = (current_layer['activations']
                                 if isinstance(current_layer['activations'], torch.Tensor)
                                 else torch.tensor(current_layer['activations']))

            # Normalize activations for visualization
            if current_activations is not None and len(current_activations) > 0:
                max_val = float(torch.max(torch.abs(current_activations))) if len(current_activations) > 0 else 1
                if max_val > 0:
                    normalized_activations = current_activations / max_val
                else:
                    normalized_activations = current_activations
            else:
                normalized_activations = torch.zeros(current_layer['size'])

            # Draw connections with varying opacity based on activation
            for j in range(current_layer['size']):
                # Only draw connections for a subset of neurons to reduce clutter
                if j % max(1, current_layer['size'] // 20) != 0:
                    continue

                start_pos = neuron_positions[i][j]

                # Calculate connection strength
                activation_strength = float(abs(normalized_activations[j])) if j < len(normalized_activations) else 0

                if activation_strength > 0.1:
                    # Connect to a subset of neurons in next layer
                    for k in range(next_layer['size']):
                        if k % max(1, next_layer['size'] // 20) != 0:
                            continue

                        end_pos = neuron_positions[i + 1][k]

                        # Calculate alpha (opacity) based on activation strength
                        alpha = min(255, int(40 + activation_strength * 215))
                        connection_color = (*self.colors['connection'][:3], alpha)

                        # Draw connection line
                        pygame.draw.line(surface, connection_color, start_pos, end_pos, 1)

        # Draw action labels
        action_labels = ['Straight', 'Turn Right', 'Turn Left']
        output_layer_x = (len(layers) - 1) * layer_width + layer_width // 2 - margin_x + 50
        for j, label in enumerate(action_labels):
            y = margin_y + j * (self.height - 2 * margin_y) // 3 + (self.height - 2 * margin_y) // 6
            font = pygame.font.Font(None, self.font_size)
            text = font.render(label, True, self.colors['text'])
            surface.blit(text, (output_layer_x, y))

class SnakeGameWithNNVisualization:
    def __init__(self, grid_size=10, game_width=600, nn_width=800):
        self.grid_size = grid_size
        self.game_width = game_width
        self.nn_width = nn_width
        self.total_width = game_width + nn_width
        self.height = 600

        # Initialize Pygame
        pygame.init()
        self.window = pygame.display.set_mode((self.total_width, self.height))
        pygame.display.set_caption("Snake DQN with Neural Network Visualization")
        self.clock = pygame.time.Clock()

        # Create surfaces
        self.game_surface = pygame.Surface((game_width, self.height))
        self.nn_surface = pygame.Surface((nn_width, self.height))

        # Initialize game
        from snake_game_objects import SnakeGameRL
        self.game = SnakeGameRL(grid_size=grid_size, display=False)

        # Initialize visualizer
        self.nn_visualizer = NeuralNetworkVisualizer(nn_width, self.height)

        # Colors for game rendering
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.GRAY = (128, 128, 128)

        self.square_size = game_width // grid_size

    def render_game(self, state_values):
        # Render the snake game on the left surface
        self.game_surface.fill(self.WHITE)

        # Draw grid lines
        for i in range(self.grid_size + 1):
            pygame.draw.line(self.game_surface, self.GRAY,
                           (i * self.square_size, 0),
                           (i * self.square_size, self.height), 1)
            pygame.draw.line(self.game_surface, self.GRAY,
                           (0, i * self.square_size),
                           (self.game_width, i * self.square_size), 1)

        # Draw food
        if self.game.food_position:
            rect = pygame.Rect(
                self.game.food_position[0] * self.square_size,
                self.game.food_position[1] * self.square_size,
                self.square_size,
                self.square_size
            )
            pygame.draw.rect(self.game_surface, self.RED, rect)
            pygame.draw.rect(self.game_surface, self.BLACK, rect, 2)

        # Draw snake
        for i, position in enumerate(self.game.snake_positions):
            rect = pygame.Rect(
                position[0] * self.square_size,
                position[1] * self.square_size,
                self.square_size,
                self.square_size
            )
            color = self.GREEN if i == 0 else self.BLACK
            pygame.draw.rect(self.game_surface, color, rect)
            pygame.draw.rect(self.game_surface, self.GRAY, rect, 2)

        # Draw game info
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {self.game.score}", True, self.BLACK)
        self.game_surface.blit(score_text, (10, 10))

        steps_text = font.render(f"Steps: {self.game.steps}", True, self.BLACK)
        self.game_surface.blit(steps_text, (10, 50))



    def run_with_agent(self, agent, num_games=5, fps=10):
        # Run the game with neural network visualization
        agent.epsilon = 0  # No exploration during visualization

        for game_num in range(num_games):
            state = self.game.reset()
            running = True

            print(f"\nStarting Game {game_num + 1}")

            while running and not self.game.done:
                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                        elif event.key == pygame.K_SPACE:
                            # Pause/resume
                            paused = True
                            while paused:
                                for pause_event in pygame.event.get():
                                    if pause_event.type == pygame.QUIT:
                                        running = False
                                        paused = False
                                    elif pause_event.type == pygame.KEYDOWN:
                                        if pause_event.key == pygame.K_SPACE:
                                            paused = False

                if not running:
                    break

                # Get action and activations from agent
                action, activations = agent.act(state, return_activations=True)

                # Take step in game
                next_state, reward, done, info = self.game.step(action)

                # Render game
                self.render_game(state)

                # Render neural network
                self.nn_visualizer.draw_network(self.nn_surface, activations, state, action)

                # Combine surfaces
                self.window.blit(self.game_surface, (0, 0))
                self.window.blit(self.nn_surface, (self.game_width, 0))

                # Draw separator line
                pygame.draw.line(self.window, (100, 100, 100),
                               (self.game_width, 0), (self.game_width, self.height), 3)

                pygame.display.flip()
                self.clock.tick(fps)

                state = next_state

            if running:
                print(f"Game {game_num + 1} finished! Score: {self.game.score}")
                # Wait a moment before next game
                pygame.time.wait(2000)

        pygame.quit()

def visualize_trained_agent(model_path, grid_size=10, num_games=3):
    # Load a trained agent and visualize it playing
    # Create agent
    agent = DQNAgent(state_size=17, action_size=3)

    # Load trained model
    try:
        agent.load(model_path)
        print(f"Loaded model from {model_path}")
    except FileNotFoundError:
        print(f"Model file {model_path} not found. Training a simple agent...")
        # Train a basic agent quickly for demonstration
        from snake_game_objects import SnakeGameRL
        game = SnakeGameRL(grid_size=grid_size, display=False)

        # Quick training
        for episode in range(100):
            state = game.reset()
            while not game.done:
                action = agent.act(state)
                next_state, reward, done, info = game.step(action)
                agent.remember(state, action, reward, next_state, done)
                state = next_state

                if len(agent.memory) > 32:
                    agent.replay(32)

        # Save the quickly trained model
        agent.save('quick_trained_model.pth')
        print("Trained and saved a basic model")

    # Create visualization
    visualizer = SnakeGameWithNNVisualization(grid_size=grid_size)

    print("Controls:")
    print("- ESC: Quit")
    print("- SPACE: Pause/Resume")
    print("\nStarting visualization...")

    # Run visualization
    visualizer.run_with_agent(agent, num_games=num_games, fps=15)

if __name__ == "__main__":
    try:
        model_path = "saved/snake_drqn_episode_500.pth"
        visualize_trained_agent(model_path, grid_size=10, num_games=3)
    except Exception as e:
        print(f"Error during visualization: {e}")