import pygame
import numpy as np
import random
from enum import Enum
from collections import deque
import math

class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)

    @staticmethod
    def get_index(direction):
        return {Direction.UP: 0, Direction.RIGHT: 1, Direction.DOWN: 2, Direction.LEFT: 3}[direction]

class SnakeGameRL:
    def __init__(self, grid_size=10, display=True, render_delay=0):
        self.grid_size = grid_size
        self.display = display
        self.render_delay = render_delay
        self.last_food_step = 0  # Track steps since last food eaten

        # Display setup
        if self.display:
            pygame.init()
            self.width = 600
            self.height = 600
            self.square_size = self.width // self.grid_size
            self.window = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Snake RL Training")
            self.clock = pygame.time.Clock()

            # Colors
            self.WHITE = (255, 255, 255)
            self.BLACK = (0, 0, 0)
            self.RED = (255, 0, 0)
            self.GREEN = (0, 255, 0)
            self.BLUE = (0, 0, 255)
            self.GRAY = (128, 128, 128)

        self.reset()

    def reset(self):
        # Snake initialization
        self.snake_positions = deque([(self.grid_size // 2, self.grid_size // 2)])
        self.direction = Direction.RIGHT

        # Food initialization
        self.food_position = self._place_food()

        # Game state
        self.score = 0
        self.steps = 0
        self.max_steps = self.grid_size * self.grid_size * 4  # Prevent infinite loops
        self.done = False
        self.won = False

        # Tracking for rewards
        self.distance_to_food = self._calculate_distance_to_food()

        return self.get_state()

    def _place_food(self):
        empty_cells = []
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if (x, y) not in self.snake_positions:
                    empty_cells.append((x, y))

        if empty_cells:
            return random.choice(empty_cells)
        return None  # Game won - no empty cells

    def _calculate_distance_to_food(self):
        head = self.snake_positions[0]
        return abs(head[0] - self.food_position[0]) + abs(head[1] - self.food_position[1])

    def get_state(self):
        state = []
        head = self.snake_positions[0]

        # Direction of food relative to head (2 values: x and y direction)
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

        state.extend([food_dir_x, food_dir_y])  # Food directions, 2 values: x and y direction

        # Current direction (4 values: one-hot encoding)
        direction_one_hot = [0, 0, 0, 0]
        direction_one_hot[Direction.get_index(self.direction)] = 1
        state.extend(direction_one_hot) # Current direction, 4 values: one-hot encoding

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

        for pos in check_positions:
            danger = 0
            if (pos[0] < 0 or pos[0] >= self.grid_size or
                pos[1] < 0 or pos[1] >= self.grid_size or
                pos in self.snake_positions):
                danger = 1
            dangers.append(danger)

        state.extend(dangers) # Danger detection, 8 values: straight, left, right, diagonals

        # Distance to food (normalized)
        dist_x = (self.food_position[0] - head[0]) / self.grid_size
        dist_y = (self.food_position[1] - head[1]) / self.grid_size
        state.extend([dist_x, dist_y]) # Distance to food, 2 values: x and y (normalized)

        # Snake length (normalized)
        state.append(len(self.snake_positions) / (self.grid_size * self.grid_size)) # Snake length, 1 value (normalized)

        # Food directions, 2 values: x and y direction
        # Current direction, 4 values: one-hot encoding
        # Danger detection, 8 values: straight, left, right, diagonals
        # Distance to food, 2 values: x and y (normalized)
        # Snake length, 1 value (normalized)

        return np.array(state, dtype=np.float32)

    def get_action_from_direction(self, new_direction):
        """Convert absolute direction to relative action (straight, left, right)"""
        current_idx = Direction.get_index(self.direction)
        new_idx = Direction.get_index(new_direction)

        # Calculate relative turn
        diff = (new_idx - current_idx) % 4

        if diff == 0:
            return 0  # Straight
        elif diff == 1:
            return 1  # Turn right
        elif diff == 3:
            return 2  # Turn left
        else:  # diff == 2 (180 degree turn - invalid)
            return 0  # Keep going straight

    def step(self, action):
        """Execute one game step with given action
        Action: 0 = straight, 1 = turn right, 2 = turn left"""

        if self.done:
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
            self.done = True
            reward = -20  # Big penalty for dying
            return self.get_state(), reward, True, {'score': self.score}

        # Move snake
        self.snake_positions.appendleft(new_head)

        # Check food collision
        if new_head == self.food_position:
            self.score += 1
            self.last_food_step = self.steps  # Reset food step counter
            reward = 30  # Big reward for eating food

            # Place new food
            self.food_position = self._place_food()
            if self.food_position is None:  # Won the game
                self.done = True
                self.won = True
                reward = 50  # Huge bonus for winning
                return self.get_state(), reward, True, {'score': self.score, 'won': True}
        else:
            # Remove tail if no food eaten
            self.snake_positions.pop()

            # Small reward/penalty based on distance to food
            new_distance = self._calculate_distance_to_food()
            if new_distance < self.distance_to_food:
                reward = 1  # Getting closer to food
            elif new_distance > self.distance_to_food:
                reward = -3  # Getting farther from food
            self.distance_to_food = new_distance

        # Check if exceeded max steps (to prevent infinite loops)
        if self.steps >= self.max_steps:
            self.done = True
            reward = -10  # Penalty for taking too long

        if self.steps - self.last_food_step >= 100:
            self.done = True
            reward = -100

        reward -= 0.01

        return self.get_state(), reward, self.done, {'score': self.score}

    def render(self):
        if not self.display:
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

        # Draw snake
        for i, position in enumerate(self.snake_positions):
            rect = pygame.Rect(
                position[0] * self.square_size,
                position[1] * self.square_size,
                self.square_size,
                self.square_size
            )
            # Head is green, body is black
            color = self.GREEN if i == 0 else self.BLACK
            pygame.draw.rect(self.window, color, rect)
            pygame.draw.rect(self.window, self.GRAY, rect, 2)

        # Draw score
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {self.score}", True, self.BLACK)
        self.window.blit(score_text, (10, 10))

        # Draw steps
        steps_text = font.render(f"Steps: {self.steps}", True, self.BLACK)
        self.window.blit(steps_text, (10, 50))

        pygame.display.flip()

        if self.render_delay > 0:
            self.clock.tick(self.render_delay)

    def close(self):
        """Close the pygame window"""
        if self.display:
            pygame.quit()

    def get_state_size(self):
        """Return the size of the state vector"""
        return 17  # 2 food dir + 4 current dir + 8 dangers + 2 distances + 1 length

class SnakeTrainer:
    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.game = SnakeGameRL(grid_size=grid_size, display=False)

    def evaluate_agent(self, agent_func, num_episodes=100, verbose=False):
        """Evaluate an agent function that takes state and returns action"""
        scores = []
        wins = 0

        for episode in range(num_episodes):
            state = self.game.reset()
            total_reward = 0

            while not self.game.done:
                action = agent_func(state)
                state, reward, done, info = self.game.step(action)
                total_reward += reward

            scores.append(self.game.score)
            if 'won' in info and info['won']:
                wins += 1

            if verbose and (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}: Score = {self.game.score}, "
                      f"Avg Score = {np.mean(scores):.2f}")

        return {
            'mean_score': np.mean(scores),
            'max_score': np.max(scores),
            'min_score': np.min(scores),
            'win_rate': wins / num_episodes,
            'scores': scores
        }

if __name__ == "__main__":
    # For human play
    game = SnakeGameRL(grid_size=10, display=True, render_delay=10)

    running = True
    state = game.reset()

    print("Use arrow keys to play. Press ESC to quit.")
    print(f"State size: {game.get_state_size()}")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_UP:
                    action = game.get_action_from_direction(Direction.UP)
                    state, reward, done, info = game.step(action)
                elif event.key == pygame.K_DOWN:
                    action = game.get_action_from_direction(Direction.DOWN)
                    state, reward, done, info = game.step(action)
                elif event.key == pygame.K_LEFT:
                    action = game.get_action_from_direction(Direction.LEFT)
                    state, reward, done, info = game.step(action)
                elif event.key == pygame.K_RIGHT:
                    action = game.get_action_from_direction(Direction.RIGHT)
                    state, reward, done, info = game.step(action)
                print("State vector:", state)

                if done:
                    print(f"Game Over! Score: {info['score']}")
                    state = game.reset()

        game.render()

    game.close()