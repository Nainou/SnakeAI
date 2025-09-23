import numpy as np
import random
from enum import Enum
from collections import deque
import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import time
import copy
import pygame

class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)

    @staticmethod
    def get_index(direction):
        return {Direction.UP: 0, Direction.RIGHT: 1, Direction.DOWN: 2, Direction.LEFT: 3}[direction]

class SnakeGameRL:
    """Snake game implementation for genetic algorithm training with optional pygame visualization"""

    def __init__(self, grid_size=10, display=False, render_delay=0):
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
            pygame.display.set_caption("Snake AI - Genetic Algorithm")
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
        self.max_steps = self.grid_size * self.grid_size * 20  # Prevent infinite loops
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

        for pos in check_positions:
            danger = 0
            if (pos[0] < 0 or pos[0] >= self.grid_size or
                pos[1] < 0 or pos[1] >= self.grid_size or
                pos in self.snake_positions):
                danger = 1
            dangers.append(danger)

        state.extend(dangers)  # Danger detection, 8 values

        # Distance to food (normalized)
        dist_x = (self.food_position[0] - head[0]) / self.grid_size
        dist_y = (self.food_position[1] - head[1]) / self.grid_size
        state.extend([dist_x, dist_y])  # Distance to food, 2 values

        # Snake length (normalized)
        state.append(len(self.snake_positions) / (self.grid_size * self.grid_size))  # 1 value

        return np.array(state, dtype=np.float32)

    def get_state_size(self):
        """Return the size of the state vector"""
        return 17  # 2 food dir + 4 current dir + 8 dangers + 2 distances + 1 length

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

        # Penalty for taking too long to find food
        if self.steps - self.last_food_step >= 100:
            self.done = True
            reward = -100

        # Small time penalty to encourage efficiency
        reward -= 0.01

        return self.get_state(), reward, self.done, {'score': self.score}

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


class NeuralNetwork(nn.Module):
    def __init__(self, input_size=17, hidden_size=64, output_size=3, device=None):
        """
        Enhanced neural network for genetic algorithm
        Larger network to break through performance plateaus
        """
        super(NeuralNetwork, self).__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Larger, deeper network for better learning capacity
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, output_size)

        # Add dropout for regularization during evolution
        self.dropout = nn.Dropout(0.1)

        # Move to device
        self.to(self.device)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class Individual:
    def __init__(self, input_size=17, hidden_size=64, output_size=3, device=None):
        """
        Represents one individual in the genetic algorithm population
        Each individual has a neural network and tracks its fitness
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = NeuralNetwork(input_size, hidden_size, output_size, self.device)
        self.fitness = 0
        self.games_played = 0
        self.total_score = 0
        self.max_score = 0
        self.total_steps = 0
        self.wins = 0

    def act(self, state):
        """Get action from the neural network"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            output = self.network(state_tensor)
            return output.argmax().item()

    def get_weights(self):
        """Extract all weights and biases as a flat numpy array"""
        weights = []
        for param in self.network.parameters():
            weights.extend(param.data.flatten().cpu().numpy())
        return np.array(weights)

    def set_weights(self, weights):
        """Set all weights and biases from a flat numpy array"""
        idx = 0
        for param in self.network.parameters():
            param_shape = param.shape
            param_size = param.numel()
            param.data = torch.FloatTensor(weights[idx:idx+param_size]).reshape(param_shape).to(self.device)
            idx += param_size

    def copy(self):
        """Create a deep copy of this individual"""
        new_individual = Individual(device=self.device)
        new_individual.network.load_state_dict(self.network.state_dict())
        new_individual.fitness = self.fitness
        return new_individual

class GeneticAlgorithm:
    def __init__(self, population_size=50, mutation_rate=0.15, crossover_rate=0.8,
                 elitism_ratio=0.15, tournament_size=5, num_threads=4, device=None):
        """
        Genetic Algorithm for evolving Snake AI

        Args:
            population_size: Number of individuals in each generation
            mutation_rate: Probability of mutation for each weight
            crossover_rate: Probability of crossover between parents
            elitism_ratio: Fraction of best individuals to keep unchanged
            tournament_size: Number of individuals in tournament selection
            num_threads: Number of threads for parallel evaluation
            device: Device to run neural networks on (cuda/cpu)
        """
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_ratio = elitism_ratio
        self.tournament_size = tournament_size
        self.num_threads = num_threads

        # Device setup
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize population
        self.population = [Individual(device=self.device) for _ in range(population_size)]

        # Statistics tracking
        self.generation = 0
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.best_score_history = []
        self.avg_score_history = []

        # Thread safety
        self.print_lock = Lock()

    def evaluate_individual_worker(self, args):
        """Worker function for threaded evaluation"""
        individual, game_class, num_games, individual_id = args

        total_score = 0
        total_steps = 0
        max_score = 0
        wins = 0

        for game_num in range(num_games):
            game = game_class(grid_size=10, display=False)
            state = game.reset()

            while not game.done:
                action = individual.act(state)
                state, reward, done, info = game.step(action)

            score = game.score
            steps = game.steps
            won = getattr(game, 'won', False)

            total_score += score
            total_steps += steps
            max_score = max(max_score, score)
            if won:
                wins += 1

        # Calculate fitness based on multiple factors
        avg_score = total_score / num_games
        avg_steps = total_steps / num_games
        win_rate = wins / num_games

        # Enhanced fitness function with non-linear rewards
        fitness = (avg_score ** 1.5 * 100 +  # Non-linear score scaling
                  max_score * 50 +           # Bonus for best single game
                  win_rate * 1500 +          # Increased bonus for winning games
                  min(avg_steps, 300) * 0.05 + # Small efficiency bonus
                  (avg_score > 15) * 200 +   # Bonus for breaking score 15
                  (avg_score > 25) * 500)    # Large bonus for breaking score 25

        individual.fitness = fitness
        individual.games_played = num_games
        individual.total_score = total_score
        individual.max_score = max_score
        individual.total_steps = total_steps
        individual.wins = wins

        return individual_id, fitness, avg_score

    def evaluate_population(self, game_class, num_games=3, verbose=False, progress_callback=None):
        """Evaluate the entire population's fitness using threading"""
        start_time = time.time()

        if verbose:
            print(f"Gen {self.generation}: Evaluating {self.population_size} individuals...")

        # Prepare arguments for threaded evaluation
        eval_args = [(individual, game_class, num_games, i)
                     for i, individual in enumerate(self.population)]

        completed = 0
        # Use ThreadPoolExecutor for parallel evaluation
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            # Submit all tasks
            futures = {executor.submit(self.evaluate_individual_worker, args): args[3]
                      for args in eval_args}

            # Process completed tasks
            for future in as_completed(futures):
                individual_id, fitness, avg_score = future.result()
                completed += 1

                if progress_callback:
                    progress_callback(completed, self.population_size)
                elif verbose and completed % max(1, self.population_size // 10) == 0:
                    with self.print_lock:
                        print(f"  Progress: {completed}/{self.population_size}")

        # Sort population by fitness (descending)
        self.population.sort(key=lambda x: x.fitness, reverse=True)

        # Update statistics
        fitnesses = [ind.fitness for ind in self.population]
        scores = [ind.total_score / ind.games_played for ind in self.population]

        self.best_fitness_history.append(fitnesses[0])
        self.avg_fitness_history.append(np.mean(fitnesses))
        self.best_score_history.append(scores[0])
        self.avg_score_history.append(np.mean(scores))

        eval_time = time.time() - start_time

        if verbose:
            print(f"Gen {self.generation}: Best={scores[0]:.1f}, Avg={np.mean(scores):.1f}, "
                  f"Fitness={fitnesses[0]:.0f} ({eval_time:.1f}s)")

    def tournament_selection(self):
        """Select an individual using tournament selection"""
        tournament = random.sample(self.population, self.tournament_size)
        return max(tournament, key=lambda x: x.fitness)

    def crossover(self, parent1, parent2):
        """Create two offspring through crossover"""
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()

        # Get parent weights
        weights1 = parent1.get_weights()
        weights2 = parent2.get_weights()

        # Single-point crossover
        crossover_point = random.randint(1, len(weights1) - 1)

        offspring1_weights = np.concatenate([weights1[:crossover_point], weights2[crossover_point:]])
        offspring2_weights = np.concatenate([weights2[:crossover_point], weights1[crossover_point:]])

        # Create offspring
        offspring1 = Individual(device=self.device)
        offspring2 = Individual(device=self.device)
        offspring1.set_weights(offspring1_weights)
        offspring2.set_weights(offspring2_weights)

        return offspring1, offspring2

    def mutate(self, individual):
        """Enhanced mutation with adaptive noise and multiple strategies"""
        weights = individual.get_weights()

        # Adaptive mutation strength based on generation
        base_strength = 0.1
        adaptive_strength = base_strength * (1.0 + 0.5 * np.exp(-self.generation / 20))

        # Apply different mutation strategies
        for i in range(len(weights)):
            if random.random() < self.mutation_rate:
                mutation_type = random.random()

                if mutation_type < 0.7:  # 70% - Gaussian noise
                    weights[i] += np.random.normal(0, adaptive_strength)
                elif mutation_type < 0.9:  # 20% - Larger jumps for exploration
                    weights[i] += np.random.normal(0, adaptive_strength * 3)
                else:  # 10% - Complete weight replacement
                    weights[i] = np.random.normal(0, 0.5)

        individual.set_weights(weights)

    def calculate_diversity(self, individual1, individual2):
        """Calculate diversity between two individuals based on weight differences"""
        weights1 = individual1.get_weights()
        weights2 = individual2.get_weights()
        return np.linalg.norm(weights1 - weights2)

    def diversity_selection(self, candidates, target_count):
        """Select individuals that maximize diversity"""
        if len(candidates) <= target_count:
            return candidates

        selected = [candidates[0]]  # Start with best individual
        remaining = candidates[1:]

        while len(selected) < target_count and remaining:
            best_candidate = None
            best_diversity = -1

            for candidate in remaining:
                min_diversity = min(self.calculate_diversity(candidate, selected_ind)
                                  for selected_ind in selected)
                if min_diversity > best_diversity:
                    best_diversity = min_diversity
                    best_candidate = candidate

            if best_candidate:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
            else:
                break

        return selected

    def evolve_generation(self):
        """Create the next generation using genetic operations with diversity preservation"""
        new_population = []

        # Enhanced elitism: keep the best individuals but ensure diversity
        elite_count = int(self.population_size * self.elitism_ratio)
        elite_candidates = self.population[:min(elite_count * 3, len(self.population))]
        elites = self.diversity_selection(elite_candidates, elite_count)
        new_population.extend([ind.copy() for ind in elites])

        # Generate offspring to fill the rest of the population
        while len(new_population) < self.population_size:
            # Inject random individuals occasionally for diversity (5% chance)
            if random.random() < 0.05 and len(new_population) < self.population_size - 1:
                random_individual = Individual(device=self.device)
                new_population.append(random_individual)
                continue

            # Select parents
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()

            # Create offspring
            offspring1, offspring2 = self.crossover(parent1, parent2)

            # Mutate offspring
            self.mutate(offspring1)
            self.mutate(offspring2)

            # Add to new population
            new_population.extend([offspring1, offspring2])

        # Ensure exact population size
        self.population = new_population[:self.population_size]
        self.generation += 1

    def get_best_individual(self):
        """Get the best individual from current population"""
        return self.population[0] if self.population else None

    def save_best(self, filepath):
        """Save the best individual's network"""
        best_individual = self.get_best_individual()
        if best_individual:
            torch.save(best_individual.network.state_dict(), filepath)

    def load_individual(self, filepath):
        """Load a neural network into a new individual"""
        individual = Individual(device=self.device)
        individual.network.load_state_dict(torch.load(filepath, map_location=self.device))
        return individual

def train_genetic_algorithm(generations=50, population_size=50, games_per_eval=3,
                          verbose=True, num_threads=4, quiet=False, device=None):
    """
    Train a genetic algorithm to play Snake

    Args:
        generations: Number of generations to evolve
        population_size: Size of population in each generation
        games_per_eval: Number of games to play for fitness evaluation
        verbose: Print detailed progress
        num_threads: Number of threads for parallel evaluation
        quiet: Minimal output mode
        device: Device to run on (cuda/cpu)
    """
    # Device setup
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize genetic algorithm
    ga = GeneticAlgorithm(population_size=population_size, num_threads=num_threads, device=device)

    if not quiet:
        print(f"🧬 Genetic Algorithm Training")
        print(f"Population: {population_size}, Generations: {generations}, Threads: {num_threads}")
        print(f"Device: {device}")
        print("=" * 60)
    else:
        print(f"🧬 Training on {device} | Pop: {population_size} | Gen: {generations} | Threads: {num_threads}")

    start_time = time.time()

    for gen in range(generations):
        gen_start = time.time()

        # Evaluate current generation
        if quiet:
            # Simple progress callback for quiet mode
            def progress_callback(completed, total):
                if completed % max(1, total // 4) == 0:
                    print(f"Gen {gen}: {completed}/{total}", end='\r')

            ga.evaluate_population(SnakeGameRL, num_games=games_per_eval,
                                 verbose=False, progress_callback=progress_callback)

            # Show generation results in quiet mode
            best_score = ga.best_score_history[-1]
            avg_score = ga.avg_score_history[-1]
            gen_time = time.time() - gen_start
            print(f"Gen {gen:2d}: Best={best_score:4.1f} Avg={avg_score:4.1f} ({gen_time:4.1f}s)")
        else:
            ga.evaluate_population(SnakeGameRL, num_games=games_per_eval, verbose=verbose)

        # Save best individual periodically
        if gen % 10 == 0:
            ga.save_best(f'genetic_snake_gen_{gen}.pth')

        # Evolve to next generation (except for the last generation)
        if gen < generations - 1:
            ga.evolve_generation()

        # Show generation summary
        gen_time = time.time() - gen_start
        if not quiet and gen % 5 == 0:
            best_score = ga.best_score_history[-1]
            avg_score = ga.avg_score_history[-1]
            print(f"Generation {gen} complete: Best={best_score:.1f}, Avg={avg_score:.1f} ({gen_time:.1f}s)")

    # Save final best individual
    ga.save_best('genetic_snake_final.pth')

    elapsed_time = time.time() - start_time

    if not quiet:
        print(f"\n✅ Training completed in {elapsed_time:.1f} seconds")
        print(f"Final best score: {ga.best_score_history[-1]:.1f}")
        print(f"Improvement: {ga.best_score_history[-1] - ga.best_score_history[0]:.1f} points")

    return ga

def plot_evolution_progress(ga):
    """Plot the evolution progress over generations"""
    plt.figure(figsize=(15, 10))

    # Plot fitness evolution
    plt.subplot(2, 2, 1)
    plt.plot(ga.best_fitness_history, label='Best Fitness', color='red', linewidth=2)
    plt.plot(ga.avg_fitness_history, label='Average Fitness', color='blue', linewidth=2)
    plt.title('Fitness Evolution')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot score evolution
    plt.subplot(2, 2, 2)
    plt.plot(ga.best_score_history, label='Best Score', color='green', linewidth=2)
    plt.plot(ga.avg_score_history, label='Average Score', color='orange', linewidth=2)
    plt.title('Score Evolution')
    plt.xlabel('Generation')
    plt.ylabel('Average Score')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot fitness distribution for last generation
    plt.subplot(2, 2, 3)
    last_gen_fitness = [ind.fitness for ind in ga.population]
    plt.hist(last_gen_fitness, bins=20, alpha=0.7, color='purple')
    plt.title(f'Fitness Distribution - Generation {ga.generation-1}')
    plt.xlabel('Fitness')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)

    # Plot score distribution for last generation
    plt.subplot(2, 2, 4)
    last_gen_scores = [ind.total_score / ind.games_played for ind in ga.population]
    plt.hist(last_gen_scores, bins=20, alpha=0.7, color='cyan')
    plt.title(f'Score Distribution - Generation {ga.generation-1}')
    plt.xlabel('Average Score')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def test_genetic_individual(model_path, num_games=10, display=True, device=None):
    """Test a saved genetic algorithm individual"""
    # Device setup
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the individual
    ga = GeneticAlgorithm(device=device)
    individual = ga.load_individual(model_path)

    # Test the individual
    game = SnakeGameRL(grid_size=10, display=display, render_delay=100 if display else 0)
    scores = []

    print(f"Testing genetic algorithm individual from {model_path}")
    print("=" * 50)

    for i in range(num_games):
        state = game.reset()

        while not game.done:
            action = individual.act(state)
            state, reward, done, info = game.step(action)

            if display:
                game.render()

                # Check for quit event
                if game.done and hasattr(game, 'window'):
                    # Game was quit via pygame window
                    break

        scores.append(game.score)
        print(f"Game {i+1}: Score = {game.score}")

        # If user closed the window, break out of the game loop
        if display and game.done and hasattr(game, 'window') and not pygame.get_init():
            break

    if display:
        game.close()

    print(f"\nTest Results:")
    print(f"Average Score: {np.mean(scores):.2f}")
    print(f"Max Score: {max(scores)}")
    print(f"Min Score: {min(scores)}")
    print(f"Standard Deviation: {np.std(scores):.2f}")

    return scores

if __name__ == "__main__":
    print("Genetic Algorithm Snake AI - Fast Training Version")
    print("=" * 50)

    # Train the genetic algorithm
    ga = train_genetic_algorithm(generations=30, population_size=30, games_per_eval=3, verbose=True)

    # Plot the evolution progress
    print("\nPlotting evolution progress...")
    plot_evolution_progress(ga)

    # Test the best individual
    print("\nTesting best individual...")
    test_scores = test_genetic_individual('genetic_snake_final.pth', num_games=5, display=False)
