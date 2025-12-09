import numpy as np
import random
import pygame
import torch
from collections import deque
from enum import Enum


# Include Snake class inline for self-contained demo
class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)

    @staticmethod
    def get_index(direction):
        return {Direction.UP: 0, Direction.RIGHT: 1, Direction.DOWN: 2, Direction.LEFT: 3}[direction]


class Snake:
    def __init__(self, snake_id, start_pos, color, grid_size, device=None):
        self.snake_id = snake_id
        self.color = color
        self.grid_size = grid_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.positions = deque([start_pos])
        self.direction = Direction.RIGHT
        self.alive = True
        self.score = 0
        self.steps = 0
        self.last_food_step = 0
        self.network = None

    def set_network(self, network):
        self.network = network

    def get_state(self, food_position, other_snakes):
        if not self.alive:
            return np.zeros(17, dtype=np.float32)
        head = self.positions[0]
        state = []
        food_dir_x = 0
        food_dir_y = 0
        if food_position[0] > head[0]:
            food_dir_x = 1
        elif food_position[0] < head[0]:
            food_dir_x = -1
        if food_position[1] > head[1]:
            food_dir_y = 1
        elif food_position[1] < head[1]:
            food_dir_y = -1
        state.extend([food_dir_x, food_dir_y])
        direction_one_hot = [0, 0, 0, 0]
        direction_one_hot[Direction.get_index(self.direction)] = 1
        state.extend(direction_one_hot)
        dangers = []
        check_positions = [
            (head[0], head[1] - 1), (head[0] + 1, head[1] - 1), (head[0] + 1, head[1]),
            (head[0] + 1, head[1] + 1), (head[0], head[1] + 1), (head[0] - 1, head[1] + 1),
            (head[0] - 1, head[1]), (head[0] - 1, head[1] - 1),
        ]
        all_occupied = set(self.positions)
        for other_snake in other_snakes:
            if other_snake.alive:
                all_occupied.update(other_snake.positions)
        for pos in check_positions:
            danger = 0
            if (pos[0] < 0 or pos[0] >= self.grid_size or pos[1] < 0 or pos[1] >= self.grid_size or pos in all_occupied):
                danger = 1
            dangers.append(danger)
        state.extend(dangers)
        dist_x = (food_position[0] - head[0]) / self.grid_size
        dist_y = (food_position[1] - head[1]) / self.grid_size
        state.extend([dist_x, dist_y])
        state.append(len(self.positions) / (self.grid_size * self.grid_size))
        return np.array(state, dtype=np.float32)

    def act(self, state, return_activations=False):
        if not self.alive or self.network is None:
            if return_activations:
                dummy_activations = {'fc1': torch.zeros(64), 'fc2': torch.zeros(64), 'fc3': torch.zeros(32), 'output': torch.zeros(3)}
                return 0, dummy_activations
            return 0
        return self.network.act(state, return_activations=return_activations)

    def move(self, action, food_position, other_snakes):
        if not self.alive:
            return False, 0
        self.steps += 1
        if action == 1:
            directions = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
            current_idx = Direction.get_index(self.direction)
            self.direction = directions[(current_idx + 1) % 4]
        elif action == 2:
            directions = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
            current_idx = Direction.get_index(self.direction)
            self.direction = directions[(current_idx - 1) % 4]
        head = self.positions[0]
        new_head = (head[0] + self.direction.value[0], head[1] + self.direction.value[1])
        reward = 0
        if (new_head[0] < 0 or new_head[0] >= self.grid_size or new_head[1] < 0 or new_head[1] >= self.grid_size):
            self.alive = False
            reward = -20
            return False, reward
        if new_head in self.positions:
            self.alive = False
            reward = -20
            return False, reward
        for other_snake in other_snakes:
            if other_snake.alive and new_head in other_snake.positions:
                self.alive = False
                reward = -20
                return False, reward
        self.positions.appendleft(new_head)
        if new_head == food_position:
            self.score += 1
            self.last_food_step = self.steps
            reward = 30
            return True, reward
        else:
            self.positions.pop()
            new_distance = abs(new_head[0] - food_position[0]) + abs(new_head[1] - food_position[1])
            old_distance = abs(head[0] - food_position[0]) + abs(head[1] - food_position[1])
            if new_distance < old_distance:
                reward = 1
            elif new_distance > old_distance:
                reward = -3
            else:
                reward = -0.01
        return True, reward


class NeuralNetworkVisualizer:
    def __init__(self, width=1200, height=800):
        self.width = width
        self.height = height
        self.colors = {
            'background': (15, 15, 25),
            'neuron_inactive': (60, 60, 80),
            'neuron_low': (100, 150, 200),
            'neuron_medium': (150, 200, 255),
            'neuron_high': (255, 200, 100),
            'neuron_very_high': (255, 100, 100),
            'connection_weak': (40, 40, 60),
            'connection_medium': (80, 100, 120),
            'connection_strong': (120, 150, 180),
            'text': (255, 255, 255),
            'text_dim': (180, 180, 180),
            'layer_bg': (25, 25, 35),
            'selected': (100, 255, 100)
        }
        self.font_size = 18
        self.small_font_size = 14
        self.label_font_size = 12

    def draw_network(self, surface, activations, state_values, action=None, hidden_size=64):
        network_rect = pygame.Rect(0, 0, self.width, self.height)
        pygame.draw.rect(surface, self.colors['background'], network_rect)

        # Layer definitions - match the PVP network architecture (17 input, 64 hidden)
        layers = [
            {'name': 'Input', 'size': 17, 'activations': state_values},
            {'name': 'Hidden 1', 'size': hidden_size, 'activations': activations.get('fc1', torch.zeros(hidden_size))},
            {'name': 'Hidden 2', 'size': hidden_size, 'activations': activations.get('fc2', torch.zeros(hidden_size))},
            {'name': 'Hidden 3', 'size': hidden_size // 2, 'activations': activations.get('fc3', torch.zeros(hidden_size // 2))},
            {'name': 'Output', 'size': 3, 'activations': activations.get('output', torch.zeros(3))}
        ]

        # Calculate layout with better spacing
        num_layers = len(layers)
        layer_width = (self.width - 100) // num_layers
        margin_x = 50
        margin_y = 60
        available_height = self.height - 2 * margin_y

        # First pass: Calculate neuron positions with better spacing
        neuron_positions = []
        for i, layer in enumerate(layers):
            positions = []
            x = margin_x + i * layer_width + layer_width // 2
            neuron_spacing = available_height / max(layer['size'] + 1, 1)

            for j in range(layer['size']):
                neuron_y = margin_y + (j + 1) * neuron_spacing
                positions.append((x, neuron_y))
            neuron_positions.append(positions)

        # Store activation data for connection drawing
        layer_activation_data = []

        # Draw layers and neurons
        for i, layer in enumerate(layers):
            x = margin_x + i * layer_width + layer_width // 2
            layer_height = available_height

            # Draw layer background
            layer_rect = pygame.Rect(margin_x + i * layer_width + 10, margin_y - 20, layer_width - 20, layer_height + 40)
            pygame.draw.rect(surface, self.colors['layer_bg'], layer_rect, border_radius=8)

            # Draw layer title
            font = pygame.font.Font(None, self.font_size)
            title_text = font.render(layer['name'], True, self.colors['text'])
            title_rect = title_text.get_rect(centerx=x, y=margin_y - 40)
            surface.blit(title_text, title_rect)

            # Define state labels for input layer (17 values for PVP)
            if layer['name'] == 'Input':
                state_labels = [
                    "Food X", "Food Y", "Dir U", "Dir R", "Dir D", "Dir L",
                    "Dng U", "Dng UR", "Dng R", "Dng DR",
                    "Dng D", "Dng DL", "Dng L", "Dng UL",
                    "Dist X", "Dist Y", "Length"
                ]
            elif layer['name'] == 'Output':
                state_labels = ['Straight', 'Turn Right', 'Turn Left']
            else:
                state_labels = None

            # Get and normalize activations
            if layer['activations'] is not None:
                if isinstance(layer['activations'], torch.Tensor):
                    layer_activations = layer['activations'].clone()
                else:
                    layer_activations = torch.tensor(layer['activations'], dtype=torch.float32)
            else:
                layer_activations = torch.zeros(layer['size'])

            # Normalize for visualization
            if len(layer_activations) > 0:
                max_val = float(torch.max(torch.abs(layer_activations)))
                if max_val > 0:
                    normalized_activations = layer_activations / max_val
                else:
                    normalized_activations = torch.zeros_like(layer_activations)
            else:
                normalized_activations = torch.zeros(layer['size'])

            layer_activation_data.append({
                'raw': layer_activations,
                'normalized': normalized_activations,
                'max_val': max_val if len(layer_activations) > 0 else 1.0
            })

            # Draw neurons
            for j, (neuron_x, neuron_y) in enumerate(neuron_positions[i]):
                # Get activation value
                if j < len(layer_activations):
                    activation = float(layer_activations[j])
                    normalized_activation = float(abs(normalized_activations[j])) if j < len(normalized_activations) else 0.0
                else:
                    activation = 0.0
                    normalized_activation = 0.0

                # Special handling for output layer
                if layer['name'] == 'Output':
                    if action is not None and j == action:
                        normalized_activation = 1.0
                        is_selected = True
                    else:
                        is_selected = False
                else:
                    is_selected = False

                # Choose color based on activation strength
                if normalized_activation > 0.8:
                    color = self.colors['neuron_very_high']
                elif normalized_activation > 0.5:
                    color = self.colors['neuron_high']
                elif normalized_activation > 0.3:
                    color = self.colors['neuron_medium']
                elif normalized_activation > 0.1:
                    color = self.colors['neuron_low']
                else:
                    color = self.colors['neuron_inactive']

                # Highlight selected output
                if is_selected:
                    color = self.colors['selected']

                # Draw neuron - scale radius based on layer size to prevent overlap
                # Smaller neurons for larger layers
                base_radius = 12
                if layer['size'] <= 20:
                    neuron_radius = base_radius
                elif layer['size'] <= 50:
                    neuron_radius = max(6, base_radius * 20 / layer['size'])
                else:
                    neuron_radius = max(4, base_radius * 20 / layer['size'])
                pygame.draw.circle(surface, color, (int(neuron_x), int(neuron_y)), int(neuron_radius))
                border_color = self.colors['selected'] if is_selected else self.colors['text']
                pygame.draw.circle(surface, border_color, (int(neuron_x), int(neuron_y)), neuron_radius, 2)

                # Draw labels
                if state_labels and j < len(state_labels):
                    label_text = state_labels[j]
                    label_font = pygame.font.Font(None, self.label_font_size)
                    label_surface = label_font.render(label_text, True, self.colors['text'])

                    # Position label to the left of input neurons, right of output
                    if layer['name'] == 'Input':
                        label_rect = label_surface.get_rect(right=neuron_x - neuron_radius - 5, centery=neuron_y)
                    elif layer['name'] == 'Output':
                        label_rect = label_surface.get_rect(left=neuron_x + neuron_radius + 5, centery=neuron_y)
                    else:
                        continue  # Don't label hidden neurons

                    surface.blit(label_surface, label_rect)

                # Draw activation value for output layer
                if layer['name'] == 'Output':
                    value_font = pygame.font.Font(None, self.small_font_size)
                    value_text = value_font.render(f'{activation:.2f}', True, self.colors['text'])
                    value_rect = value_text.get_rect(centerx=neuron_x, y=neuron_y + neuron_radius + 8)
                    surface.blit(value_text, value_rect)

        # Draw connections - only show strongest connections to reduce clutter
        max_connections_per_neuron = 5  # Only show top 5 connections per neuron
        connection_threshold = 0.2  # Minimum activation to show connection

        for i in range(len(layers) - 1):
            current_layer = layers[i]
            next_layer = layers[i + 1]
            current_activations = layer_activation_data[i]['normalized']

            # For each neuron in current layer, find strongest connections
            for j in range(current_layer['size']):
                if j >= len(current_activations):
                    continue

                activation_strength = float(abs(current_activations[j]))
                if activation_strength < connection_threshold:
                    continue

                start_pos = neuron_positions[i][j]

                # Get next layer activations
                next_activations = layer_activation_data[i + 1]['normalized']

                # Calculate connection strengths (simplified - using activations)
                # In reality, we'd need weights, but this gives a visual sense
                connection_strengths = []
                for k in range(next_layer['size']):
                    if k < len(next_activations):
                        # Combined strength based on both activations
                        combined_strength = activation_strength * float(abs(next_activations[k]))
                        connection_strengths.append((k, combined_strength))

                # Sort by strength and take top N
                connection_strengths.sort(key=lambda x: x[1], reverse=True)
                top_connections = connection_strengths[:max_connections_per_neuron]

                # Draw top connections
                for k, strength in top_connections:
                    if strength < connection_threshold * 0.5:
                        continue

                    end_pos = neuron_positions[i + 1][k]

                    # Choose connection color and width based on strength
                    if strength > 0.6:
                        # Strong connection - bright and thick
                        intensity = min(255, int(150 + strength * 105))
                        conn_color = (intensity, intensity, intensity)
                        line_width = 2
                    elif strength > 0.3:
                        # Medium connection
                        intensity = min(200, int(100 + strength * 100))
                        conn_color = (intensity, intensity, intensity)
                        line_width = 1
                    else:
                        # Weak connection - dim
                        intensity = int(60 + strength * 40)
                        conn_color = (intensity, intensity, intensity)
                        line_width = 1

                    # Draw connection line
                    pygame.draw.line(surface, conn_color, start_pos, end_pos, line_width)


class SnakeGamePvP:
    # PvP Snake game implementation
    def __init__(self, grid_size=20, num_snakes=2, display=False, render_delay=0):
        self.grid_size = grid_size
        self.num_snakes = min(max(1, num_snakes), 4)  # 1-4 snakes (1 for single player, 2-4 for PvP)
        self.display = display
        self.render_delay = render_delay

        # Snake colors (distinct colors for each snake)
        self.snake_colors = [
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 0, 0),    # Red
            (255, 255, 0)   # Yellow
        ]

        # Display setup
        if self.display:
            pygame.init()
            self.game_width = 800
            self.nn_width = 800  # Neural network visualization width
            self.width = self.game_width + self.nn_width
            self.height = 800
            self.square_size = min(self.game_width // self.grid_size, self.height // self.grid_size)
            self.window = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("PvP Snake AI - Genetic Algorithm")
            self.clock = pygame.time.Clock()

            # Create surfaces for game and neural network
            self.game_surface = pygame.Surface((self.game_width, self.height))
            self.nn_surface = pygame.Surface((self.nn_width, self.height))

            # Initialize neural network visualizer
            self.nn_visualizer = NeuralNetworkVisualizer(self.nn_width, self.height)

            # Selected snake for network visualization
            self.selected_snake_id = None
            # Store last state and activations for visualization
            self.last_state = None
            self.last_activations = None
            self.last_action = None

            # Colors
            self.WHITE = (255, 255, 255)
            self.BLACK = (0, 0, 0)
            self.RED = (255, 0, 0)
            self.GRAY = (128, 128, 128)
            self.LIGHT_GREEN = (170, 215, 81)
            self.DARK_GREEN = (162, 209, 73)

            # Rendering parameters
            self.segment_margin = 5
            self.food_margin = 7
            self.segment_width = self.square_size - 2 * self.segment_margin

            # Score font
            self.score_font = pygame.font.SysFont("None", 36)

        self.reset()

    def reset(self):
        # Initialize snakes at different starting positions
        start_positions = self._get_start_positions()

        # Preserve existing networks if they exist
        old_networks = {}
        if hasattr(self, 'snakes') and self.snakes:
            for snake in self.snakes:
                if hasattr(snake, 'network') and snake.network is not None:
                    old_networks[snake.snake_id] = snake.network

        self.snakes = []

        for i in range(self.num_snakes):
            snake = Snake(
                snake_id=i,
                start_pos=start_positions[i],
                color=self.snake_colors[i],
                grid_size=self.grid_size
            )
            # Restore network if it existed
            if i in old_networks:
                snake.set_network(old_networks[i])
            self.snakes.append(snake)

        # Food initialization
        self.food_position = self._place_food()

        # Game state
        self.steps = 0
        self.max_steps = self.grid_size * self.grid_size * 10  # Prevent infinite loops
        self.done = False
        self.winner = None

        return self.get_states()

    def _get_start_positions(self):
        # Get starting positions for snakes (spread out)
        positions = []
        center = self.grid_size // 2

        if self.num_snakes == 1:
            positions = [
                (center, center)
            ]
        elif self.num_snakes == 2:
            positions = [
                (center - 2, center),
                (center + 2, center)
            ]
        elif self.num_snakes == 3:
            positions = [
                (center - 3, center),
                (center + 3, center),
                (center, center - 3)
            ]
        elif self.num_snakes == 4:
            positions = [
                (center - 3, center - 3),
                (center + 3, center - 3),
                (center - 3, center + 3),
                (center + 3, center + 3)
            ]

        return positions

    def _place_food(self):
        # Place food in an empty cell
        empty_cells = []
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                # Check if cell is empty (not occupied by any snake)
                occupied = False
                for snake in self.snakes:
                    if snake.alive and (x, y) in snake.positions:
                        occupied = True
                        break
                if not occupied:
                    empty_cells.append((x, y))

        if empty_cells:
            return random.choice(empty_cells)
        return None  # No empty cells - game should end

    def get_states(self):
        # Get states for all alive snakes
        states = {}
        for snake in self.snakes:
            if snake.alive:
                other_snakes = [s for s in self.snakes if s.snake_id != snake.snake_id]
                states[snake.snake_id] = snake.get_state(self.food_position, other_snakes)
        return states

    def step(self, actions):
        # Execute one game step with given actions for all snakes
        # Args:
        #   actions: Dict mapping snake_id to action (0=straight, 1=right, 2=left)
        if self.done:
            return self.get_states(), {}, True, {}

        self.steps += 1
        rewards = {}

        # Store state and activations for selected snake if displaying
        if self.display and self.selected_snake_id is not None:
            selected_snake = next((s for s in self.snakes if s.snake_id == self.selected_snake_id), None)
            if selected_snake and selected_snake.alive and selected_snake.network is not None:
                other_snakes = [s for s in self.snakes if s.snake_id != selected_snake.snake_id]
                state = selected_snake.get_state(self.food_position, other_snakes)
                action, activations = selected_snake.act(state, return_activations=True)
                self.last_state = state
                self.last_activations = activations
                self.last_action = actions.get(self.selected_snake_id, action)

        # Move all alive snakes
        for snake in self.snakes:
            if snake.alive and snake.snake_id in actions:
                other_snakes = [s for s in self.snakes if s.snake_id != snake.snake_id]
                moved, reward = snake.move(actions[snake.snake_id], self.food_position, other_snakes)
                rewards[snake.snake_id] = reward

                # Check if snake ate food
                if moved and reward == 30:  # Food eaten
                    # Place new food
                    self.food_position = self._place_food()
                    if self.food_position is None:  # No more empty cells
                        self.done = True
                        self.winner = snake.snake_id
                        break

        # Check if only one snake is alive (winner)
        alive_snakes = [s for s in self.snakes if s.alive]
        if len(alive_snakes) <= 1:
            self.done = True
            if alive_snakes:
                self.winner = alive_snakes[0].snake_id

        # Check if exceeded max steps
        if self.steps >= self.max_steps:
            self.done = True
            # Winner is snake with highest score
            if self.snakes:
                self.winner = max(self.snakes, key=lambda s: s.score).snake_id

        # Check for snakes that haven't eaten food in too long
        for snake in self.snakes:
            if snake.alive and self.steps - snake.last_food_step >= 100:
                snake.alive = False
                rewards[snake.snake_id] = -100

        return self.get_states(), rewards, self.done, {'winner': self.winner, 'scores': {s.snake_id: s.score for s in self.snakes}}

    def set_snake_networks(self, networks):
        # Set neural networks for snakes
        # Args:
        #   networks: Dict mapping snake_id to neural network
        for snake_id, network in networks.items():
            if snake_id < len(self.snakes):
                self.snakes[snake_id].set_network(network)

    def get_alive_snakes(self):
        # Get list of alive snakes
        return [s for s in self.snakes if s.alive]

    def get_snake_scores(self):
        # Get scores for all snakes
        return {s.snake_id: s.score for s in self.snakes}

    def handle_mouse_click(self, mouse_x, mouse_y):
        # Handle mouse click for selecting snakes
        # Check if click is on a snake name (in the game area)
        if mouse_x < self.game_width:
            y_offset = 10
            for snake in self.snakes:
                # Use the same dimensions as the actual text background
                # The background is drawn at (6, y_offset) with size (text_width + 16, text_height + 8)
                text_width, text_height = self.score_font.size(f"Snake {snake.snake_id}: {snake.score}")
                text_rect = pygame.Rect(6, y_offset, text_width + 16, text_height + 8)
                if text_rect.collidepoint(mouse_x, mouse_y):
                    # Toggle selection
                    if self.selected_snake_id == snake.snake_id:
                        self.selected_snake_id = None
                    else:
                        self.selected_snake_id = snake.snake_id
                        # Get current state and activations
                        if snake.alive and snake.network is not None:
                            other_snakes = [s for s in self.snakes if s.snake_id != snake.snake_id]
                            state = snake.get_state(self.food_position, other_snakes)
                            action, activations = snake.act(state, return_activations=True)
                            self.last_state = state
                            self.last_activations = activations
                            self.last_action = action
                    return True
                y_offset += 35
        return False

    def render(self):
        # Render the game using pygame if display is enabled
        if not self.display:
            return

        # Handle pygame events (only QUIT here, mouse clicks handled in demo loop)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.done = True
                return

        # Render game on left surface
        self.game_surface.fill(self.WHITE)

        # Draw checkerboard background
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                color = self.LIGHT_GREEN if (x + y) % 2 == 0 else self.DARK_GREEN
                pygame.draw.rect(self.game_surface, color,
                               (x * self.square_size, y * self.square_size,
                                self.square_size, self.square_size))

        # Draw food
        if self.food_position:
            rect = pygame.Rect(
                self.food_position[0] * self.square_size + self.food_margin,
                self.food_position[1] * self.square_size + self.food_margin,
                self.square_size - self.food_margin * 2,
                self.square_size - self.food_margin * 2
            )
            pygame.draw.rect(self.game_surface, self.RED, rect)

        # Draw snakes with detailed rendering
        for snake in self.snakes:
            if not snake.alive:
                continue

            positions = list(snake.positions)
            for i, position in enumerate(positions):
                x = position[0] * self.square_size
                y = position[1] * self.square_size
                prev = positions[i - 1] if i > 0 else None

                # Use snake color, slightly darker for body
                if i == 0:  # Head
                    color = snake.color
                else:  # Body
                    color = tuple(max(0, c - 50) for c in snake.color)

                # Draw connection to previous segment
                if prev:
                    # Calculate direction to previous segment
                    dx = prev[0] - position[0]
                    dy = prev[1] - position[1]

                    # Handle screen wrap-around
                    if abs(dx) > 1:  # Wrapped horizontally
                        dx = -1 if dx > 0 else 1
                    if abs(dy) > 1:  # Wrapped vertically
                        dy = -1 if dy > 0 else 1

                    # Draw connecting rectangle
                    if dx != 0:  # Horizontal connection
                        connect_x = x + self.segment_margin
                        if dx < 0:  # Going left
                            connect_x = x - (self.square_size - self.segment_margin * 2)
                        if dx > 0:  # Going right
                            connect_x = x + (self.square_size - self.segment_margin * 2)
                        connect_y = y + self.segment_margin
                        connect_w = self.square_size - self.segment_margin * 2 + 5
                        connect_h = self.segment_width
                        pygame.draw.rect(self.window, color,
                                       (connect_x, connect_y, connect_w, connect_h))
                    elif dy != 0:  # Vertical connection
                        connect_x = x + self.segment_margin
                        connect_y = y + self.segment_margin
                        if dy < 0:  # Going up
                            connect_y = y - (self.square_size - self.segment_margin * 2)
                        if dy > 0:  # Going down
                            connect_y = y + (self.square_size - self.segment_margin * 2)
                        connect_w = self.segment_width
                        connect_h = self.square_size - self.segment_margin * 2 + 5
                        pygame.draw.rect(self.game_surface, color,
                                       (connect_x, connect_y, connect_w, connect_h))

                # Draw segment square
                rect = pygame.Rect(x + self.segment_margin, y + self.segment_margin,
                                 self.segment_width, self.segment_width)
                pygame.draw.rect(self.game_surface, color, rect)

            # Draw tongue and eyes on head
            if positions:
                head = positions[0]
                hx, hy = head[0] * self.square_size, head[1] * self.square_size

                # Get direction from snake's direction attribute
                dx, dy = snake.direction.value

                # Draw tongue
                tongue_color = (255, 0, 0)
                tongue_length = 8
                tongue_width = 2
                fork_size = 2
                if dx == 1:  # Right
                    pygame.draw.rect(self.game_surface, tongue_color,
                                   (hx - 5 + self.square_size, hy + self.square_size//2 - tongue_width//2,
                                    tongue_length, tongue_width))
                    pygame.draw.rect(self.game_surface, tongue_color,
                                   (hx - 5 + self.square_size + tongue_length,
                                    hy + self.square_size//2 - tongue_width//2 - fork_size,
                                    fork_size, fork_size))
                    pygame.draw.rect(self.game_surface, tongue_color,
                                   (hx - 5 + self.square_size + tongue_length,
                                    hy + self.square_size//2 - tongue_width//2 + tongue_width,
                                    fork_size, fork_size))
                elif dx == -1:  # Left
                    pygame.draw.rect(self.game_surface, tongue_color,
                                   (hx + 5 - tongue_length, hy + self.square_size//2 - tongue_width//2,
                                    tongue_length, tongue_width))
                    pygame.draw.rect(self.game_surface, tongue_color,
                                   (hx + 5 - tongue_length - fork_size,
                                    hy + self.square_size//2 - tongue_width//2 - fork_size,
                                    fork_size, fork_size))
                    pygame.draw.rect(self.game_surface, tongue_color,
                                   (hx + 5 - tongue_length - fork_size,
                                    hy + self.square_size//2 - tongue_width//2 + tongue_width,
                                    fork_size, fork_size))
                elif dy == 1:  # Down
                    pygame.draw.rect(self.game_surface, tongue_color,
                                   (hx + self.square_size//2 - tongue_width//2,
                                    hy - 5 + self.square_size, tongue_width, tongue_length))
                    pygame.draw.rect(self.game_surface, tongue_color,
                                   (hx + self.square_size//2 - tongue_width//2 - fork_size,
                                    hy - 5 + self.square_size + tongue_length,
                                    fork_size, fork_size))
                    pygame.draw.rect(self.game_surface, tongue_color,
                                   (hx + self.square_size//2 - tongue_width//2 + tongue_width,
                                    hy - 5 + self.square_size + tongue_length,
                                    fork_size, fork_size))
                else:  # Up
                    pygame.draw.rect(self.game_surface, tongue_color,
                                   (hx + self.square_size//2 - tongue_width//2,
                                    hy + 5 - tongue_length, tongue_width, tongue_length))
                    pygame.draw.rect(self.game_surface, tongue_color,
                                   (hx + self.square_size//2 - tongue_width//2 - fork_size,
                                    hy + 5 - tongue_length - fork_size,
                                    fork_size, fork_size))
                    pygame.draw.rect(self.game_surface, tongue_color,
                                   (hx + self.square_size//2 - tongue_width//2 + tongue_width,
                                    hy + 5 - tongue_length - fork_size,
                                    fork_size, fork_size))

                # Draw eyes
                eye_color = self.BLACK
                eye_size = 4
                if dx == 1:  # Right
                    pygame.draw.rect(self.game_surface, eye_color,
                                   (hx + 5 + self.square_size//2,
                                    hy + self.segment_margin + 5, eye_size, eye_size))
                    pygame.draw.rect(self.game_surface, eye_color,
                                   (hx + 5 + self.square_size//2,
                                    hy + self.square_size - self.segment_margin - 5 - eye_size,
                                    eye_size, eye_size))
                elif dx == -1:  # Left
                    pygame.draw.rect(self.game_surface, eye_color,
                                   (hx + 5 + self.square_size//2 - eye_size,
                                    hy + self.segment_margin + 5, eye_size, eye_size))
                    pygame.draw.rect(self.game_surface, eye_color,
                                   (hx + 5 + self.square_size//2 - eye_size,
                                    hy + self.square_size - self.segment_margin - 5 - eye_size,
                                    eye_size, eye_size))
                elif dy == 1:  # Down
                    pygame.draw.rect(self.game_surface, eye_color,
                                   (hx + self.segment_margin + 5,
                                    hy + 5 + self.square_size//2, eye_size, eye_size))
                    pygame.draw.rect(self.game_surface, eye_color,
                                   (hx + self.square_size - self.segment_margin - 5 - eye_size,
                                    hy + 5 + self.square_size//2, eye_size, eye_size))
                else:  # Up
                    pygame.draw.rect(self.game_surface, eye_color,
                                   (hx + self.segment_margin + 5,
                                    hy + 5 + self.square_size//2 - eye_size, eye_size, eye_size))
                    pygame.draw.rect(self.game_surface, eye_color,
                                   (hx + self.square_size - self.segment_margin - 5 - eye_size,
                                    hy + 5 + self.square_size//2 - eye_size, eye_size, eye_size))

        # Draw scores with semi-transparent background (clickable)
        y_offset = 10
        for snake in self.snakes:
            # Highlight if selected
            if self.selected_snake_id == snake.snake_id:
                highlight_color = (255, 255, 0, 150)
            else:
                highlight_color = (0, 0, 0, 100)

            score_text = self.score_font.render(f"Snake {snake.snake_id}: {snake.score}", True, snake.color)
            score_bg = pygame.Surface((score_text.get_width() + 16, score_text.get_height() + 8), pygame.SRCALPHA)
            score_bg.fill(highlight_color)
            self.game_surface.blit(score_bg, (6, y_offset))
            self.game_surface.blit(score_text, (14, y_offset + 4))
            y_offset += 35

        # Draw steps
        steps_text = self.score_font.render(f"Steps: {self.steps}", True, self.BLACK)
        steps_bg = pygame.Surface((steps_text.get_width() + 16, steps_text.get_height() + 8), pygame.SRCALPHA)
        steps_bg.fill((0, 0, 0, 100))
        self.game_surface.blit(steps_bg, (6, y_offset))
        self.game_surface.blit(steps_text, (14, y_offset + 4))

        # Draw winner if game is done
        if self.done and self.winner is not None:
            winner_text = self.score_font.render(f"Winner: Snake {self.winner}!", True, self.BLACK)
            winner_bg = pygame.Surface((winner_text.get_width() + 16, winner_text.get_height() + 8), pygame.SRCALPHA)
            winner_bg.fill((255, 255, 0, 200))
            self.game_surface.blit(winner_bg, (6, y_offset + 40))
            self.game_surface.blit(winner_text, (14, y_offset + 44))

        # Draw neural network visualization on right side if snake is selected
        if self.selected_snake_id is not None and self.last_state is not None and self.last_activations is not None:
            selected_snake = next((s for s in self.snakes if s.snake_id == self.selected_snake_id), None)
            if selected_snake and selected_snake.network is not None:
                # Get hidden size from network
                hidden_size = selected_snake.network.fc1.out_features
                self.nn_visualizer.draw_network(
                    self.nn_surface,
                    self.last_activations,
                    self.last_state,
                    self.last_action,
                    hidden_size=hidden_size
                )
            else:
                # Draw placeholder if snake has no network
                self.nn_surface.fill(self.nn_visualizer.colors['background'])
                font = pygame.font.Font(None, 36)
                text = font.render("No network available", True, self.nn_visualizer.colors['text'])
                text_rect = text.get_rect(center=(self.nn_width // 2, self.height // 2))
                self.nn_surface.blit(text, text_rect)
        else:
            # Draw placeholder when no snake is selected
            self.nn_surface.fill(self.nn_visualizer.colors['background'])
            font = pygame.font.Font(None, 36)
            text = font.render("Click on a snake name to view its network", True, self.nn_visualizer.colors['text'])
            text_rect = text.get_rect(center=(self.nn_width // 2, self.height // 2))
            self.nn_surface.blit(text, text_rect)

        # Combine surfaces
        self.window.blit(self.game_surface, (0, 0))
        self.window.blit(self.nn_surface, (self.game_width, 0))

        # Draw separator line
        pygame.draw.line(self.window, (100, 100, 100),
                       (self.game_width, 0), (self.game_width, self.height), 3)

        pygame.display.flip()

        # Always tick the clock to prevent freezing, use at least 1 FPS
        if self.render_delay > 0:
            self.clock.tick(self.render_delay)
        else:
            self.clock.tick(1)  # Minimum 1 FPS to prevent freezing

    def close(self):
        # Close the pygame window
        if self.display:
            pygame.quit()
