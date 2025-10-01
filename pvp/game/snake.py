"""
Individual Snake class for PvP Snake game.

Each snake has its own position, direction, color, and neural network.
"""

import numpy as np
from enum import Enum
from collections import deque
import torch
import torch.nn as nn


class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)

    @staticmethod
    def get_index(direction):
        return {Direction.UP: 0, Direction.RIGHT: 1, Direction.DOWN: 2, Direction.LEFT: 3}[direction]


class Snake:
    """Individual snake in the PvP game"""

    def __init__(self, snake_id, start_pos, color, grid_size, device=None):
        self.snake_id = snake_id
        self.color = color
        self.grid_size = grid_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Snake state
        self.positions = deque([start_pos])
        self.direction = Direction.RIGHT
        self.alive = True
        self.score = 0
        self.steps = 0
        self.last_food_step = 0

        # Neural network for AI
        self.network = None

    def set_network(self, network):
        """Set the neural network for this snake"""
        self.network = network

    def get_state(self, food_position, other_snakes):
        """Get the current state for this snake's neural network"""
        if not self.alive:
            return np.zeros(17, dtype=np.float32)

        head = self.positions[0]
        state = []

        # Direction of food relative to head (2 values: x and y direction)
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

        state.extend([food_dir_x, food_dir_y])  # Food directions, 2 values

        # Current direction (4 values: one-hot encoding)
        direction_one_hot = [0, 0, 0, 0]
        direction_one_hot[Direction.get_index(self.direction)] = 1
        state.extend(direction_one_hot)  # Current direction, 4 values

        # Danger detection in 8 directions (straight, left, right, diagonals)
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

        # Collect all occupied positions (this snake + other snakes)
        all_occupied = set(self.positions)
        for other_snake in other_snakes:
            if other_snake.alive:
                all_occupied.update(other_snake.positions)

        for pos in check_positions:
            danger = 0
            if (pos[0] < 0 or pos[0] >= self.grid_size or
                pos[1] < 0 or pos[1] >= self.grid_size or
                pos in all_occupied):
                danger = 1
            dangers.append(danger)

        state.extend(dangers)  # Danger detection, 8 values

        # Distance to food (normalized)
        dist_x = (food_position[0] - head[0]) / self.grid_size
        dist_y = (food_position[1] - head[1]) / self.grid_size
        state.extend([dist_x, dist_y])  # Distance to food, 2 values

        # Snake length (normalized)
        state.append(len(self.positions) / (self.grid_size * self.grid_size))  # 1 value

        return np.array(state, dtype=np.float32)

    def act(self, state):
        """Get action from the neural network"""
        if not self.alive or self.network is None:
            return 0  # Default action: go straight

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            output = self.network(state_tensor)
            return output.argmax().item()

    def move(self, action, food_position, other_snakes):
        """Move the snake based on action and check for collisions"""
        if not self.alive:
            return False, 0

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
        head = self.positions[0]
        new_head = (
            head[0] + self.direction.value[0],
            head[1] + self.direction.value[1]
        )

        # Check collisions
        reward = 0

        # Check wall collision
        if (new_head[0] < 0 or new_head[0] >= self.grid_size or
            new_head[1] < 0 or new_head[1] >= self.grid_size):
            self.alive = False
            reward = -20  # Big penalty for dying
            return False, reward

        # Check self collision
        if new_head in self.positions:
            self.alive = False
            reward = -20  # Big penalty for dying
            return False, reward

        # Check collision with other snakes
        for other_snake in other_snakes:
            if other_snake.alive and new_head in other_snake.positions:
                self.alive = False
                reward = -20  # Big penalty for dying
                return False, reward

        # Move snake
        self.positions.appendleft(new_head)

        # Check food collision
        if new_head == food_position:
            self.score += 1
            self.last_food_step = self.steps
            reward = 30  # Big reward for eating food
            return True, reward
        else:
            # Remove tail if no food eaten
            self.positions.pop()

            # Small reward/penalty based on distance to food
            new_distance = abs(new_head[0] - food_position[0]) + abs(new_head[1] - food_position[1])
            old_distance = abs(head[0] - food_position[0]) + abs(head[1] - food_position[1])

            if new_distance < old_distance:
                reward = 1  # Getting closer to food
            elif new_distance > old_distance:
                reward = -3  # Getting farther from food
            else:
                reward = -0.01  # Small time penalty

        return True, reward

    def reset(self, start_pos):
        """Reset snake to initial state"""
        self.positions = deque([start_pos])
        self.direction = Direction.RIGHT
        self.alive = True
        self.score = 0
        self.steps = 0
        self.last_food_step = 0
