import pygame
import numpy as np
import random
from enum import Enum
from collections import deque
import torch


class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)

    @staticmethod
    def get_index(direction):
        return {Direction.UP: 0, Direction.RIGHT: 1, Direction.DOWN: 2, Direction.LEFT: 3}[direction]


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

    def draw_network(self, surface, activations, state_values, action=None, hidden_size=64, input_size=17):
        network_rect = pygame.Rect(0, 0, self.width, self.height)
        pygame.draw.rect(surface, self.colors['background'], network_rect)

        # Determine layer structure based on input size
        if input_size == 17:
            # PvP model: 17 input
            input_labels = [
                "Food X", "Food Y", "Dir U", "Dir R", "Dir D", "Dir L",
                "Dng U", "Dng UR", "Dng R", "Dng DR",
                "Dng D", "Dng DL", "Dng L", "Dng UL",
                "Dist X", "Dist Y", "Length"
            ]
        else:
            # Ray model: 21 input
            input_labels = [f"Input {i+1}" for i in range(input_size)]

        layers = [
            {'name': 'Input', 'size': input_size, 'activations': state_values, 'labels': input_labels},
            {'name': 'Hidden 1', 'size': hidden_size, 'activations': activations.get('fc1', torch.zeros(hidden_size))},
            {'name': 'Hidden 2', 'size': hidden_size, 'activations': activations.get('fc2', torch.zeros(hidden_size))},
            {'name': 'Hidden 3', 'size': hidden_size // 2, 'activations': activations.get('fc3', torch.zeros(hidden_size // 2))},
            {'name': 'Output', 'size': 3, 'activations': activations.get('output', torch.zeros(3)), 'labels': ['Straight', 'Turn Right', 'Turn Left']}
        ]

        # Calculate layout
        num_layers = len(layers)
        layer_width = (self.width - 100) // num_layers
        margin_x = 50
        margin_y = 60
        available_height = self.height - 2 * margin_y

        # Calculate neuron positions
        neuron_positions = []
        for i, layer in enumerate(layers):
            positions = []
            x = margin_x + i * layer_width + layer_width // 2
            neuron_spacing = available_height / max(layer['size'] + 1, 1)

            for j in range(layer['size']):
                neuron_y = margin_y + (j + 1) * neuron_spacing
                positions.append((x, neuron_y))
            neuron_positions.append(positions)

        # Store activation data
        layer_activation_data = []

        # Draw layers and neurons
        for i, layer in enumerate(layers):
            x = margin_x + i * layer_width + layer_width // 2

            # Draw layer background
            layer_rect = pygame.Rect(margin_x + i * layer_width + 10, margin_y - 20, layer_width - 20, available_height + 40)
            pygame.draw.rect(surface, self.colors['layer_bg'], layer_rect, border_radius=8)

            # Draw layer title
            font = pygame.font.Font(None, self.font_size)
            title_text = font.render(layer['name'], True, self.colors['text'])
            title_rect = title_text.get_rect(centerx=x, y=margin_y - 40)
            surface.blit(title_text, title_rect)

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

                # Choose color based on activation
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
                if 'labels' in layer and layer['labels'] and j < len(layer['labels']):
                    label_text = layer['labels'][j]
                    label_font = pygame.font.Font(None, self.label_font_size)
                    label_surface = label_font.render(label_text, True, self.colors['text'])

                    if layer['name'] == 'Input':
                        label_rect = label_surface.get_rect(right=neuron_x - neuron_radius - 5, centery=neuron_y)
                    elif layer['name'] == 'Output':
                        label_rect = label_surface.get_rect(left=neuron_x + neuron_radius + 5, centery=neuron_y)
                    else:
                        continue

                    surface.blit(label_surface, label_rect)

                # Draw activation value for output layer
                if layer['name'] == 'Output':
                    value_font = pygame.font.Font(None, self.small_font_size)
                    value_text = value_font.render(f'{activation:.2f}', True, self.colors['text'])
                    value_rect = value_text.get_rect(centerx=neuron_x, y=neuron_y + neuron_radius + 8)
                    surface.blit(value_text, value_rect)

        # Draw connections
        max_connections_per_neuron = 5
        connection_threshold = 0.2

        for i in range(len(layers) - 1):
            current_layer = layers[i]
            next_layer = layers[i + 1]
            current_activations = layer_activation_data[i]['normalized']

            for j in range(current_layer['size']):
                if j >= len(current_activations):
                    continue

                activation_strength = float(abs(current_activations[j]))
                if activation_strength < connection_threshold:
                    continue

                start_pos = neuron_positions[i][j]
                next_activations = layer_activation_data[i + 1]['normalized']

                connection_strengths = []
                for k in range(next_layer['size']):
                    if k < len(next_activations):
                        combined_strength = activation_strength * float(abs(next_activations[k]))
                        connection_strengths.append((k, combined_strength))

                connection_strengths.sort(key=lambda x: x[1], reverse=True)
                top_connections = connection_strengths[:max_connections_per_neuron]

                for k, strength in top_connections:
                    if strength < connection_threshold * 0.5:
                        continue

                    end_pos = neuron_positions[i + 1][k]

                    if strength > 0.6:
                        intensity = min(255, int(150 + strength * 105))
                        conn_color = (intensity, intensity, intensity)
                        line_width = 2
                    elif strength > 0.3:
                        intensity = min(200, int(100 + strength * 100))
                        conn_color = (intensity, intensity, intensity)
                        line_width = 1
                    else:
                        intensity = int(60 + strength * 40)
                        conn_color = (intensity, intensity, intensity)
                        line_width = 1

                    pygame.draw.line(surface, conn_color, start_pos, end_pos, line_width)

    def draw_pixel_network(self, surface, activations, action=None):
        # Draw the ConvQN network for pixel models
        network_rect = pygame.Rect(0, 0, self.width, self.height)
        pygame.draw.rect(surface, self.colors['background'], network_rect)

        # Pixel model architecture:
        # Conv layers -> Pooled features (64) -> Head input (64+8=72) -> Hidden (128) -> Output (3)
        layers = [
            {'name': 'Conv Features', 'size': 64, 'activations': activations.get('conv', torch.zeros(64))},
            {'name': 'Head Input', 'size': 72, 'activations': activations.get('head_input', torch.zeros(72))},
            {'name': 'Head Hidden', 'size': 128, 'activations': activations.get('head_hidden', torch.zeros(128))},
            {'name': 'Output', 'size': 3, 'activations': activations.get('output', torch.zeros(3)),
             'labels': ['Straight', 'Turn Right', 'Turn Left']}
        ]

        # Calculate layout
        num_layers = len(layers)
        layer_width = (self.width - 100) // num_layers
        margin_x = 50
        margin_y = 60
        available_height = self.height - 2 * margin_y

        # Calculate neuron positions
        neuron_positions = []
        for i, layer in enumerate(layers):
            positions = []
            x = margin_x + i * layer_width + layer_width // 2
            neuron_spacing = available_height / max(layer['size'] + 1, 1)

            for j in range(layer['size']):
                neuron_y = margin_y + (j + 1) * neuron_spacing
                positions.append((x, neuron_y))
            neuron_positions.append(positions)

        # Store activation data
        layer_activation_data = []

        # Draw layers and neurons
        for i, layer in enumerate(layers):
            x = margin_x + i * layer_width + layer_width // 2

            # Draw layer background
            layer_rect = pygame.Rect(margin_x + i * layer_width + 10, margin_y - 20, layer_width - 20, available_height + 40)
            pygame.draw.rect(surface, self.colors['layer_bg'], layer_rect, border_radius=8)

            # Draw layer title
            font = pygame.font.Font(None, self.font_size)
            title_text = font.render(layer['name'], True, self.colors['text'])
            title_rect = title_text.get_rect(centerx=x, y=margin_y - 40)
            surface.blit(title_text, title_rect)

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

                # Choose color based on activation
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

                if is_selected:
                    color = self.colors['selected']

                # Draw neuron - scale radius based on layer size to prevent overlap
                # Smaller neurons for larger layers
                base_radius = 10
                if layer['size'] <= 20:
                    neuron_radius = base_radius
                elif layer['size'] <= 50:
                    neuron_radius = max(5, base_radius * 20 / layer['size'])
                else:
                    neuron_radius = max(3, base_radius * 20 / layer['size'])
                pygame.draw.circle(surface, color, (int(neuron_x), int(neuron_y)), int(neuron_radius))
                border_color = self.colors['selected'] if is_selected else self.colors['text']
                pygame.draw.circle(surface, border_color, (int(neuron_x), int(neuron_y)), neuron_radius, 2)

                # Draw labels for output layer
                if 'labels' in layer and layer['labels'] and j < len(layer['labels']):
                    label_text = layer['labels'][j]
                    label_font = pygame.font.Font(None, self.label_font_size)
                    label_surface = label_font.render(label_text, True, self.colors['text'])
                    label_rect = label_surface.get_rect(left=neuron_x + neuron_radius + 5, centery=neuron_y)
                    surface.blit(label_surface, label_rect)

                # Draw activation value for output layer
                if layer['name'] == 'Output':
                    value_font = pygame.font.Font(None, self.small_font_size)
                    value_text = value_font.render(f'{activation:.2f}', True, self.colors['text'])
                    value_rect = value_text.get_rect(centerx=neuron_x, y=neuron_y + neuron_radius + 8)
                    surface.blit(value_text, value_rect)

        # Draw connections (simplified - show top connections)
        max_connections_per_neuron = 5
        connection_threshold = 0.2

        for i in range(len(layers) - 1):
            current_layer = layers[i]
            next_layer = layers[i + 1]
            current_activations = layer_activation_data[i]['normalized']

            for j in range(min(current_layer['size'], len(current_activations))):
                activation_strength = float(abs(current_activations[j]))
                if activation_strength < connection_threshold:
                    continue

                start_pos = neuron_positions[i][j]
                next_activations = layer_activation_data[i + 1]['normalized']

                connection_strengths = []
                for k in range(min(next_layer['size'], len(next_activations))):
                    combined_strength = activation_strength * float(abs(next_activations[k]))
                    connection_strengths.append((k, combined_strength))

                connection_strengths.sort(key=lambda x: x[1], reverse=True)
                top_connections = connection_strengths[:max_connections_per_neuron]

                for k, strength in top_connections:
                    if strength < connection_threshold * 0.5:
                        continue

                    end_pos = neuron_positions[i + 1][k]

                    if strength > 0.6:
                        intensity = min(255, int(150 + strength * 105))
                        conn_color = (intensity, intensity, intensity)
                        line_width = 2
                    elif strength > 0.3:
                        intensity = min(200, int(100 + strength * 100))
                        conn_color = (intensity, intensity, intensity)
                        line_width = 1
                    else:
                        intensity = int(60 + strength * 40)
                        conn_color = (intensity, intensity, intensity)
                        line_width = 1

                    pygame.draw.line(surface, conn_color, start_pos, end_pos, line_width)


class SnakeGameSingle:
    def __init__(self, grid_size=20, display=True, render_delay=10, state_size=21, use_pixel_state=False):
        self.grid_size = grid_size
        self.display = display
        self.render_delay = render_delay
        self.state_size = state_size  # 17 for PvP models, 21 for Ray models
        self.use_pixel_state = use_pixel_state  # True for pixel models
        self.last_food_step = 0

        # Display setup
        if self.display:
            pygame.init()
            self.game_width = 800
            self.nn_width = 800
            self.width = self.game_width + self.nn_width
            self.height = 800
            self.square_size = min(self.game_width // self.grid_size, self.height // self.grid_size)
            self.window = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Snake AI Demo - Single Player")
            self.clock = pygame.time.Clock()

            # Create surfaces
            self.game_surface = pygame.Surface((self.game_width, self.height))
            self.nn_surface = pygame.Surface((self.nn_width, self.height))

            # Initialize neural network visualizer
            self.nn_visualizer = NeuralNetworkVisualizer(self.nn_width, self.height)

            # Store state and activations for visualization
            self.last_state = None
            self.last_activations = None
            self.last_action = None

            # Colors
            self.WHITE = (255, 255, 255)
            self.BLACK = (0, 0, 0)
            self.RED = (255, 0, 0)
            self.GREEN = (0, 255, 0)
            self.BLUE = (0, 0, 255)
            self.GRAY = (128, 128, 128)
            self.LIGHT_GREEN = (170, 215, 81)
            self.DARK_GREEN = (162, 209, 73)

            # Rendering parameters
            self.segment_margin = 5
            self.food_margin = 7
            self.segment_width = self.square_size - 2 * self.segment_margin

            # Score font
            self.score_font = pygame.font.SysFont("None", 36)

        # Neural network
        self.network = None

        self.reset()

    def set_network(self, network):
        self.network = network

    def reset(self):
        center = self.grid_size // 2
        self.snake_positions = deque([(center, center)])
        self.direction = Direction.RIGHT
        self.alive = True
        self.score = 0
        self.steps = 0
        self.max_steps = self.grid_size * self.grid_size * 20
        self.done = False
        self.won = False
        self.food_position = self._place_food()
        self.distance_to_food = self._calculate_distance_to_food()
        if self.use_pixel_state:
            return self.get_pixel_state()
        else:
            return self.get_state()

    def _place_food(self):
        empty_cells = []
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if (x, y) not in self.snake_positions:
                    empty_cells.append((x, y))

        if empty_cells:
            return random.choice(empty_cells)
        return None

    def _calculate_distance_to_food(self):
        head = self.snake_positions[0]
        return abs(head[0] - self.food_position[0]) + abs(head[1] - self.food_position[1])

    def get_state(self):
        if not self.alive:
            return np.zeros(self.state_size, dtype=np.float32)

        head = self.snake_positions[0]
        state = []

        # Direction of food relative to head (2 values)
        food_dir_x = 0
        food_dir_y = 0
        if self.food_position[0] > head[0]:
            food_dir_x = 1
        elif self.food_position[0] < head[0]:
            food_dir_x = -1
        if self.food_position[1] > head[1]:
            food_dir_y = 1
        elif self.food_position[1] < head[1]:
            food_dir_y = -1

        state.extend([food_dir_x, food_dir_y])

        # Current direction (4 values: one-hot encoding)
        direction_one_hot = [0, 0, 0, 0]
        direction_one_hot[Direction.get_index(self.direction)] = 1
        state.extend(direction_one_hot)

        # Danger detection in 8 directions
        dangers = []
        check_positions = [
            (head[0], head[1] - 1),  # Up
            (head[0] + 1, head[1] - 1),  # Up-Right
            (head[0] + 1, head[1]),  # Right
            (head[0] + 1, head[1] + 1),  # Down-Right
            (head[0], head[1] + 1),  # Down
            (head[0] - 1, head[1] + 1),  # Down-Left
            (head[0] - 1, head[1]),  # Left
            (head[0] - 1, head[1] - 1),  # Up-Left
        ]

        for pos in check_positions:
            danger = 0
            if (pos[0] < 0 or pos[0] >= self.grid_size or
                pos[1] < 0 or pos[1] >= self.grid_size or
                pos in self.snake_positions):
                danger = 1
            dangers.append(danger)

        state.extend(dangers)  # 8 values

        # Distance to food (normalized)
        dist_x = (self.food_position[0] - head[0]) / self.grid_size
        dist_y = (self.food_position[1] - head[1]) / self.grid_size
        state.extend([dist_x, dist_y])  # 2 values

        # Snake length (normalized)
        state.append(len(self.snake_positions) / (self.grid_size * self.grid_size))  # 1 value

        # For 21-input models (Ray), add tail direction (4 values)
        if self.state_size == 21:
            tail_direction_one_hot = [0, 0, 0, 0]
            if len(self.snake_positions) >= 2:
                tail = self.snake_positions[-1]
                second_to_last = self.snake_positions[-2]
                tail_dx = tail[0] - second_to_last[0]
                tail_dy = tail[1] - second_to_last[1]

                if tail_dx == 0 and tail_dy == -1:
                    tail_direction = Direction.UP
                elif tail_dx == 0 and tail_dy == 1:
                    tail_direction = Direction.DOWN
                elif tail_dx == -1 and tail_dy == 0:
                    tail_direction = Direction.LEFT
                elif tail_dx == 1 and tail_dy == 0:
                    tail_direction = Direction.RIGHT
                else:
                    tail_direction = self.direction

                tail_direction_one_hot[Direction.get_index(tail_direction)] = 1
            else:
                tail_direction_one_hot[Direction.get_index(self.direction)] = 1

            state.extend(tail_direction_one_hot)  # 4 values

        return np.array(state, dtype=np.float32)

    def get_pixel_state(self):
        # Return pixel model state format: tuple of (planes, direction)
        if not self.alive:
            # Return zero state
            H = W = self.grid_size
            planes = torch.zeros((5, H, W), dtype=torch.float32)
            direction = torch.zeros(8, dtype=torch.float32)
            return (planes, direction)

        H = W = self.grid_size

        grid_snake_head = torch.zeros((H, W), dtype=torch.float32)
        grid_snake_body = torch.zeros((H, W), dtype=torch.float32)
        grid_snake_tail = torch.zeros((H, W), dtype=torch.float32)
        grid_food_position = torch.zeros((H, W), dtype=torch.float32)
        grid_wall = torch.zeros((H, W), dtype=torch.float32)

        # Mark snake positions
        positions = list(self.snake_positions)
        if len(positions) > 1:
            for x, y in positions[1:-1]:
                grid_snake_body[y, x] = 1.0

        # Mark snake head
        if positions:
            grid_snake_head[positions[0][1], positions[0][0]] = 1.0

        # Mark snake tail
        if len(positions) > 1:
            grid_snake_tail[positions[-1][1], positions[-1][0]] = 1.0

        # Mark food position
        if self.food_position:
            fx, fy = self.food_position
            grid_food_position[fy, fx] = 1.0

        # Walls as a border
        grid_wall[0, :] = 1.0
        grid_wall[-1, :] = 1.0
        grid_wall[:, 0] = 1.0
        grid_wall[:, -1] = 1.0

        # Stack maps together for conv2d layer
        state_maps = torch.stack([grid_snake_head, grid_snake_body, grid_snake_tail,
                                  grid_food_position, grid_wall], dim=0)

        # Get direction encoding
        idx = Direction.get_index(self.direction)
        dir_onehot = torch.zeros(4, dtype=torch.float32)
        dir_onehot[idx] = 1.0

        # Tail end direction (4 values: one-hot encoding)
        tail_dir_one_hot = torch.zeros(4, dtype=torch.float32)
        if len(self.snake_positions) >= 2:
            tail = self.snake_positions[-1]
            second_to_last = self.snake_positions[-2]
            tail_dx = tail[0] - second_to_last[0]
            tail_dy = tail[1] - second_to_last[1]

            if tail_dx == 0 and tail_dy == -1:
                tail_direction = Direction.UP
            elif tail_dx == 0 and tail_dy == 1:
                tail_direction = Direction.DOWN
            elif tail_dx == -1 and tail_dy == 0:
                tail_direction = Direction.LEFT
            elif tail_dx == 1 and tail_dy == 0:
                tail_direction = Direction.RIGHT
            else:
                tail_direction = self.direction

            tail_dir_one_hot[Direction.get_index(tail_direction)] = 1.0
        else:
            tail_dir_one_hot[Direction.get_index(self.direction)] = 1.0

        extra_information = torch.cat([dir_onehot, tail_dir_one_hot])

        return (state_maps, extra_information)

    def step(self, action):
        if self.done or not self.alive:
            if self.use_pixel_state:
                return self.get_pixel_state(), 0, True, {}
            else:
                return self.get_state(), 0, True, {}

        self.steps += 1

        # Update direction based on action
        if action == 1:  # Turn right
            directions = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
            current_idx = Direction.get_index(self.direction)
            self.direction = directions[(current_idx + 1) % 4]
        elif action == 2:  # Turn left
            directions = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
            current_idx = Direction.get_index(self.direction)
            self.direction = directions[(current_idx - 1) % 4]
        # action == 0: keep going straight

        # Move snake
        head = self.snake_positions[0]
        new_head = (
            head[0] + self.direction.value[0],
            head[1] + self.direction.value[1]
        )

        # Check collisions
        reward = 0
        if (new_head[0] < 0 or new_head[0] >= self.grid_size or
            new_head[1] < 0 or new_head[1] >= self.grid_size or
            new_head in self.snake_positions):
            self.alive = False
            self.done = True
            reward = -20
            if self.use_pixel_state:
                return self.get_pixel_state(), reward, True, {'score': self.score}
            else:
                return self.get_state(), reward, True, {'score': self.score}

        # Move snake
        self.snake_positions.appendleft(new_head)

        # Check food collision
        if new_head == self.food_position:
            self.score += 1
            self.last_food_step = self.steps
            reward = 30
            self.food_position = self._place_food()
            if self.food_position is None:  # Won the game
                self.done = True
                self.won = True
                reward = 50
                if self.use_pixel_state:
                    return self.get_pixel_state(), reward, True, {'score': self.score, 'won': True}
                else:
                    return self.get_state(), reward, True, {'score': self.score, 'won': True}
        else:
            # Remove tail if no food eaten
            self.snake_positions.pop()

            # Small reward/penalty based on distance to food
            new_distance = self._calculate_distance_to_food()
            if new_distance < self.distance_to_food:
                reward = 1
            elif new_distance > self.distance_to_food:
                reward = -3
            self.distance_to_food = new_distance

        # Check if exceeded max steps
        if self.steps >= self.max_steps:
            self.done = True
            reward = -10

        if self.steps - self.last_food_step >= 100:
            self.done = True
            reward = -100

        reward -= 0.01

        if self.use_pixel_state:
            return self.get_pixel_state(), reward, self.done, {'score': self.score}
        else:
            return self.get_state(), reward, self.done, {'score': self.score}

    def render(self):
        if not self.display:
            return

        # Handle pygame events
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

        # Draw snake
        if self.alive:
            positions = list(self.snake_positions)
            for i, position in enumerate(positions):
                x = position[0] * self.square_size
                y = position[1] * self.square_size
                prev = positions[i - 1] if i > 0 else None

                # Use uniform green color for entire snake
                snake_color = self.GREEN  # Base color for the snake
                color = snake_color  # Same color for head and body

                # Draw connection to previous segment (use same color for uniformity)
                if prev:
                    # Use same color for connections
                    prev_color = snake_color

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
                        if dx < 0:  # Going left - connection extends left from current segment
                            connect_x = x - (self.square_size - self.segment_margin * 2)
                            connect_y = y + self.segment_margin
                            connect_w = self.square_size - self.segment_margin * 2 + 5
                            connect_h = self.segment_width
                        else:  # dx > 0, Going right - connection extends right from current segment
                            connect_x = x + self.square_size - self.segment_margin
                            connect_y = y + self.segment_margin
                            connect_w = self.square_size - self.segment_margin * 2 + 5
                            connect_h = self.segment_width
                        pygame.draw.rect(self.game_surface, prev_color,
                                       (connect_x, connect_y, connect_w, connect_h))
                    elif dy != 0:  # Vertical connection
                        connect_x = x + self.segment_margin
                        if dy < 0:  # Going up - connection extends up from current segment
                            connect_y = y - (self.square_size - self.segment_margin * 2)
                            connect_w = self.segment_width
                            connect_h = self.square_size - self.segment_margin * 2 + 5
                        else:  # dy > 0, Going down - connection extends down from current segment
                            connect_y = y + self.square_size - self.segment_margin
                            connect_w = self.segment_width
                            connect_h = self.square_size - self.segment_margin * 2 + 5
                        pygame.draw.rect(self.game_surface, prev_color,
                                       (connect_x, connect_y, connect_w, connect_h))

                # Draw segment
                rect = pygame.Rect(x + self.segment_margin, y + self.segment_margin,
                                 self.segment_width, self.segment_width)
                pygame.draw.rect(self.game_surface, color, rect)

            # Draw tongue and eyes on head
            if positions:
                head = positions[0]
                hx, hy = head[0] * self.square_size, head[1] * self.square_size

                # Get direction from snake's direction attribute
                dx, dy = self.direction.value

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

        # Draw score
        score_text = self.score_font.render(f"Score: {self.score}", True, self.BLACK)
        score_bg = pygame.Surface((score_text.get_width() + 16, score_text.get_height() + 8), pygame.SRCALPHA)
        score_bg.fill((0, 0, 0, 100))
        self.game_surface.blit(score_bg, (6, 6))
        self.game_surface.blit(score_text, (14, 10))

        # Draw steps
        steps_text = self.score_font.render(f"Steps: {self.steps}", True, self.BLACK)
        steps_bg = pygame.Surface((steps_text.get_width() + 16, steps_text.get_height() + 8), pygame.SRCALPHA)
        steps_bg.fill((0, 0, 0, 100))
        self.game_surface.blit(steps_bg, (6, 50))
        self.game_surface.blit(steps_text, (14, 54))

        # Draw game over message
        if self.done and not self.alive:
            game_over_text = self.score_font.render("Game Over!", True, self.BLACK)
            game_over_bg = pygame.Surface((game_over_text.get_width() + 16, game_over_text.get_height() + 8), pygame.SRCALPHA)
            game_over_bg.fill((255, 0, 0, 200))
            self.game_surface.blit(game_over_bg, (6, 94))
            self.game_surface.blit(game_over_text, (14, 98))

        # Draw neural network visualization on right side
        if self.network is not None and self.last_activations is not None:
            # Visualize based on network type
            if hasattr(self.network, 'fc1') and self.last_state is not None:
                # FC network (Ray/PvP models)
                hidden_size = self.network.fc1.out_features
                self.nn_visualizer.draw_network(
                    self.nn_surface,
                    self.last_activations,
                    self.last_state,
                    self.last_action,
                    hidden_size=hidden_size,
                    input_size=self.state_size
                )
            elif hasattr(self.network, 'convLayers'):
                # Pixel model (ConvQN)
                self.nn_visualizer.draw_pixel_network(
                    self.nn_surface,
                    self.last_activations,
                    self.last_action
                )
            else:
                # Fallback placeholder
                self.nn_surface.fill(self.nn_visualizer.colors['background'])
                font = pygame.font.Font(None, 36)
                text = font.render("Neural Network", True, self.nn_visualizer.colors['text'])
                text_rect = text.get_rect(center=(self.nn_width // 2, self.height // 2))
                self.nn_surface.blit(text, text_rect)
        else:
            # Draw placeholder
            self.nn_surface.fill(self.nn_visualizer.colors['background'])
            font = pygame.font.Font(None, 36)
            text = font.render("Neural Network Visualization", True, self.nn_visualizer.colors['text'])
            text_rect = text.get_rect(center=(self.nn_width // 2, self.height // 2))
            self.nn_surface.blit(text, text_rect)

        # Combine surfaces
        self.window.blit(self.game_surface, (0, 0))
        self.window.blit(self.nn_surface, (self.game_width, 0))

        # Draw separator line
        pygame.draw.line(self.window, (100, 100, 100),
                       (self.game_width, 0), (self.game_width, self.height), 3)

        pygame.display.flip()

        if self.render_delay > 0:
            self.clock.tick(self.render_delay)

    def close(self):
        if self.display:
            pygame.quit()
