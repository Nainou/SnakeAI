import numpy as np
import random
import torch
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from .neural_network import NeuralNetwork


class Individual:
    # Represents one individual in the genetic algorithm population

    def __init__(self, input_size=17, hidden_size=64, output_size=3, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = NeuralNetwork(input_size, hidden_size, output_size, self.device)
        self.fitness = 0
        self.games_played = 0
        self.total_score = 0
        self.max_score = 0
        self.total_steps = 0
        self.wins = 0
        self.avg_position = 0  # Average finishing position in PvP games

    def act(self, state):
        # Get action from the neural network
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            output = self.network(state_tensor)
            return output.argmax().item()

    def get_weights(self):
        # Extract all weights and biases as a flat numpy array
        weights = []
        for param in self.network.parameters():
            weights.extend(param.data.flatten().cpu().numpy())
        return np.array(weights)

    def set_weights(self, weights):
        # Set all weights and biases from a flat numpy array
        idx = 0
        for param in self.network.parameters():
            param_shape = param.shape
            param_size = param.numel()
            param.data = torch.FloatTensor(weights[idx:idx+param_size]).reshape(param_shape).to(self.device)
            idx += param_size

    def copy(self):
        # Create a deep copy of this individual
        new_individual = Individual(device=self.device)
        new_individual.network.load_state_dict(self.network.state_dict())
        new_individual.fitness = self.fitness
        return new_individual


class GeneticAlgorithm:
    # Genetic Algorithm for evolving PvP Snake AI

    def __init__(self, population_size=50, mutation_rate=0.15, crossover_rate=0.8,
                 elitism_ratio=0.15, tournament_size=5, num_threads=4, device=None):
        # Initialize the genetic algorithm
        # Args:
        #   population_size: Number of individuals in each generation
        #   mutation_rate: Probability of mutation for each weight
        #   crossover_rate: Probability of crossover between parents
        #   elitism_ratio: Fraction of best individuals to keep unchanged
        #   tournament_size: Number of individuals in tournament selection
        #   num_threads: Number of threads for parallel evaluation
        #   device: Device to run neural networks on (cuda/cpu)
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
        # Worker function for threaded evaluation of PvP games
        individual, game_class, num_games, individual_id, num_snakes = args

        total_score = 0
        total_steps = 0
        max_score = 0
        wins = 0
        total_position = 0

        for game_num in range(num_games):
            # Create PvP game
            game = game_class(grid_size=20, num_snakes=num_snakes, display=False)

            # Set up all snakes with random opponents (including self)
            networks = {}
            for i in range(num_snakes):
                if i == 0:  # First snake is always the individual being evaluated
                    networks[i] = individual.network
                else:
                    # Use random individual from population as opponent
                    opponent = random.choice(self.population)
                    networks[i] = opponent.network

            game.set_snake_networks(networks)
            states = game.reset()

            # Play the game
            while not game.done:
                actions = {}
                for snake_id, state in states.items():
                    if snake_id in networks:
                        actions[snake_id] = networks[snake_id].act(state)

                states, rewards, done, info = game.step(actions)

            # Calculate results for this individual
            snake_scores = game.get_snake_scores()
            winner = info.get('winner')

            # Find this individual's snake
            individual_snake_id = None
            for snake_id, network in networks.items():
                if network == individual.network:
                    individual_snake_id = snake_id
                    break

            if individual_snake_id is not None:
                score = snake_scores.get(individual_snake_id, 0)
                steps = game.steps
                won = (winner == individual_snake_id)

                total_score += score
                total_steps += steps
                max_score = max(max_score, score)
                if won:
                    wins += 1

                # Calculate position (1st, 2nd, etc.)
                sorted_scores = sorted(snake_scores.values(), reverse=True)
                position = sorted_scores.index(score) + 1
                total_position += position

        # Calculate fitness based on multiple factors
        avg_score = total_score / num_games
        avg_steps = total_steps / num_games
        win_rate = wins / num_games
        avg_position = total_position / num_games

        # Enhanced fitness function for PvP
        fitness = (avg_score ** 1.5 * 100 +           # Non-linear score scaling
                  max_score * 50 +                    # Bonus for best single game
                  win_rate * 2000 +                   # High bonus for winning
                  (4 - avg_position) * 500 +          # Position bonus (1st=3, 2nd=2, etc.)
                  min(avg_steps, 300) * 0.05 +        # Small efficiency bonus
                  (avg_score > 10) * 300 +            # Bonus for breaking score 10
                  (avg_score > 20) * 800)             # Large bonus for breaking score 20

        individual.fitness = fitness
        individual.games_played = num_games
        individual.total_score = total_score
        individual.max_score = max_score
        individual.total_steps = total_steps
        individual.wins = wins
        individual.avg_position = avg_position

        return individual_id, fitness, avg_score, avg_position

    def evaluate_population(self, game_class, num_games=3, num_snakes=2, verbose=False, progress_callback=None):
        # Evaluate the entire population's fitness using threading
        start_time = time.time()

        if verbose:
            print(f"Gen {self.generation}: Evaluating {self.population_size} individuals...")

        # Prepare arguments for threaded evaluation
        eval_args = [(individual, game_class, num_games, i, num_snakes)
                     for i, individual in enumerate(self.population)]

        completed = 0
        # Use ThreadPoolExecutor for parallel evaluation
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            # Submit all tasks
            futures = {executor.submit(self.evaluate_individual_worker, args): args[2]
                      for args in eval_args}

            # Process completed tasks
            for future in as_completed(futures):
                individual_id, fitness, avg_score, avg_position = future.result()
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
                  f"Fitness={fitnesses[0]:.0f}, Pos={self.population[0].avg_position:.1f} ({eval_time:.1f}s)")

    def tournament_selection(self):
        # Select an individual using tournament selection
        tournament = random.sample(self.population, self.tournament_size)
        return max(tournament, key=lambda x: x.fitness)

    def crossover(self, parent1, parent2):
        # Create two offspring through crossover
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
        # Enhanced mutation with adaptive noise
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

    def evolve_generation(self):
        # Create the next generation using genetic operations
        new_population = []

        # Elitism: keep the best individuals
        elite_count = int(self.population_size * self.elitism_ratio)
        elites = self.population[:elite_count]
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
        # Get the best individual from current population
        return self.population[0] if self.population else None

    def save_best(self, filepath, include_metadata=True):
        # Save the best individual's network
        # Args:
        #   filepath: Path to save the model
        #   include_metadata: If True, include architecture metadata in filename
        best_individual = self.get_best_individual()
        if best_individual:
            # Extract architecture info from network
            if include_metadata:
                input_size = best_individual.network.fc1.in_features
                hidden_size = best_individual.network.fc1.out_features
                output_size = best_individual.network.fc4.out_features

                # Try to parse existing filename for extra info
                from pathlib import Path
                import re
                import sys

                # Try to import metadata utilities (may not be available)
                try:
                    sys.path.append(str(Path(__file__).parent.parent.parent))
                    from demo.demo import parse_model_filename, create_model_filename

                    path_obj = Path(filepath)
                    existing_meta = parse_model_filename(path_obj.name)
                    extra_info = existing_meta.extra_info if existing_meta else ""

                    # If no extra info and filename contains 'gen' or 'final', extract it
                    if not extra_info:
                        if 'gen' in path_obj.stem.lower():
                            gen_match = re.search(r'gen[_\s]*(\d+)', path_obj.stem.lower())
                            if gen_match:
                                extra_info = f"gen{gen_match.group(1)}"
                        elif 'final' in path_obj.stem.lower():
                            extra_info = "final"

                    # Create new filename with metadata
                    new_filename = create_model_filename(
                        model_type="pvp",
                        input_size=input_size,
                        hidden_size=hidden_size,
                        output_size=output_size,
                        extra_info=extra_info
                    )
                    filepath = path_obj.parent / new_filename
                except ImportError:
                    # If metadata module not available, just save with original filename
                    pass

            torch.save(best_individual.network.state_dict(), filepath)

    def load_individual(self, filepath):
        # Load a neural network into a new individual
        individual = Individual(device=self.device)
        individual.network.load_state_dict(torch.load(filepath, map_location=self.device))
        return individual
