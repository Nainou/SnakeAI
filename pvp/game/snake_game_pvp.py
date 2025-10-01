"""
PvP Snake Game Implementation

Supports 2-4 snakes competing on a large grid with genetic algorithm AI.
Each snake has distinct colors and can use different saved models.
"""

import numpy as np
import random
import pygame
import torch
from collections import deque
from .snake import Snake, Direction


class SnakeGamePvP:
    """PvP Snake game implementation for genetic algorithm training with optional pygame visualization"""

    def __init__(self, grid_size=20, num_snakes=2, display=False, render_delay=0):
        self.grid_size = grid_size
        self.num_snakes = min(max(2, num_snakes), 4)  # 2-4 snakes
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
            self.width = 800
            self.height = 800
            self.square_size = min(self.width // self.grid_size, self.height // self.grid_size)
            self.window = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("PvP Snake AI - Genetic Algorithm")
            self.clock = pygame.time.Clock()

            # Colors
            self.WHITE = (255, 255, 255)
            self.BLACK = (0, 0, 0)
            self.RED = (255, 0, 0)
            self.GRAY = (128, 128, 128)

        self.reset()

    def reset(self):
        """Reset the game with new snake positions"""
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
        """Get starting positions for snakes (spread out)"""
        positions = []
        center = self.grid_size // 2

        if self.num_snakes == 2:
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
        """Place food in an empty cell"""
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
        """Get states for all alive snakes"""
        states = {}
        for snake in self.snakes:
            if snake.alive:
                other_snakes = [s for s in self.snakes if s.snake_id != snake.snake_id]
                states[snake.snake_id] = snake.get_state(self.food_position, other_snakes)
        return states

    def step(self, actions):
        """Execute one game step with given actions for all snakes

        Args:
            actions: Dict mapping snake_id to action (0=straight, 1=right, 2=left)
        """
        if self.done:
            return self.get_states(), {}, True, {}

        self.steps += 1
        rewards = {}

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
        """Set neural networks for snakes

        Args:
            networks: Dict mapping snake_id to neural network
        """
        for snake_id, network in networks.items():
            if snake_id < len(self.snakes):
                self.snakes[snake_id].set_network(network)

    def get_alive_snakes(self):
        """Get list of alive snakes"""
        return [s for s in self.snakes if s.alive]

    def get_snake_scores(self):
        """Get scores for all snakes"""
        return {s.snake_id: s.score for s in self.snakes}

    def render(self):
        """Render the game using pygame if display is enabled"""
        if not self.display:
            return

        # Handle pygame events to prevent window from becoming unresponsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.done = True
                return

        self.window.fill(self.WHITE)

        # Draw grid lines
        for i in range(self.grid_size + 1):
            pygame.draw.line(self.window, self.GRAY,
                           (i * self.square_size, 0),
                           (i * self.square_size, self.height), 1)
            pygame.draw.line(self.window, self.GRAY,
                           (0, i * self.square_size),
                           (self.width, i * self.square_size), 1)

        # Draw food
        if self.food_position:
            rect = pygame.Rect(
                self.food_position[0] * self.square_size,
                self.food_position[1] * self.square_size,
                self.square_size,
                self.square_size
            )
            pygame.draw.rect(self.window, self.RED, rect)
            pygame.draw.rect(self.window, self.BLACK, rect, 2)

        # Draw snakes
        for snake in self.snakes:
            if not snake.alive:
                continue

            for i, position in enumerate(snake.positions):
                rect = pygame.Rect(
                    position[0] * self.square_size,
                    position[1] * self.square_size,
                    self.square_size,
                    self.square_size
                )
                # Head is brighter, body is darker
                if i == 0:  # Head
                    color = snake.color
                else:  # Body
                    color = tuple(max(0, c - 50) for c in snake.color)

                pygame.draw.rect(self.window, color, rect)
                pygame.draw.rect(self.window, self.BLACK, rect, 1)

        # Draw scores
        font = pygame.font.Font(None, 24)
        y_offset = 10
        for snake in self.snakes:
            score_text = font.render(f"Snake {snake.snake_id}: {snake.score}", True, snake.color)
            self.window.blit(score_text, (10, y_offset))
            y_offset += 25

        # Draw steps
        steps_text = font.render(f"Steps: {self.steps}", True, self.BLACK)
        self.window.blit(steps_text, (10, y_offset))

        # Draw winner if game is done
        if self.done and self.winner is not None:
            winner_text = font.render(f"Winner: Snake {self.winner}!", True, self.BLACK)
            self.window.blit(winner_text, (10, y_offset + 30))

        pygame.display.flip()

        if self.render_delay > 0:
            self.clock.tick(self.render_delay)

    def close(self):
        """Close the pygame window"""
        if self.display:
            pygame.quit()
