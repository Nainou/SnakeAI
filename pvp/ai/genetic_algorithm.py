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
        new_individual = Individual(input_size=17, hidden_size=64, output_size=3, device=self.device)
        new_individual.network.load_state_dict(self.network.state_dict())
        new_individual.fitness = self.fitness
        return new_individual


class GeneticAlgorithm:
    # Genetic Algorithm for evolving PvP Snake AI

    def __init__(self, population_size=50, mutation_rate=0.05, crossover_rate=0.8,
                 elitism_ratio=0.15, tournament_size=5, num_threads=4, device=None,
                 selection_type='plus', crossover_type='mixed', num_parents=None, num_offspring=None):
        # Initialize the genetic algorithm
        # Args:
        #   population_size: Number of individuals in each generation (used if num_parents/num_offspring not set)
        #   mutation_rate: Probability of mutation for each weight (static 5%)
        #   crossover_rate: Probability of crossover between parents
        #   elitism_ratio: Fraction of best individuals to keep unchanged
        #   tournament_size: Number of individuals in tournament selection
        #   num_threads: Number of threads for parallel evaluation
        #   device: Device to run neural networks on (cuda/cpu)
        #   selection_type: 'plus' for (μ+λ) selection, 'tournament' for tournament selection
        #   crossover_type: 'mixed' for 50% SBX + 50% SPBX, 'sbx' for SBX only, 'spbx' for SPBX only
        #   num_parents: Number of parents (if None, uses population_size)
        #   num_offspring: Number of offspring to generate (if None, uses population_size)

        # Use num_parents/num_offspring if provided, otherwise use population_size
        if num_parents is None:
            num_parents = population_size
        if num_offspring is None:
            num_offspring = population_size

        self.num_parents = num_parents
        self.num_offspring = num_offspring
        self.population_size = num_parents  # Final population size is num_parents
        self.mutation_rate = mutation_rate  # Static 5%
        self.crossover_rate = crossover_rate
        self.elitism_ratio = elitism_ratio
        self.tournament_size = tournament_size
        self.num_threads = num_threads
        self.selection_type = selection_type
        self.crossover_type = crossover_type

        # Device setup
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize population with PvP architecture (17, 64, 3)
        self.population = [Individual(input_size=17, hidden_size=64, output_size=3, device=self.device) for _ in range(num_parents)]

        # Statistics tracking
        self.generation = 0
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.best_score_history = []
        self.avg_score_history = []

        # Thread safety
        self.print_lock = Lock()

    def evaluate_game_worker(self, args):
        # Worker function for coevolution: evaluates a game between multiple individuals
        # All individuals in the game get fitness updates (true coevolution)
        game_class, individual_indices, num_games, num_snakes = args

        # Get the individuals for this game
        game_individuals = [self.population[idx] for idx in individual_indices]

        # Initialize tracking for all individuals in this game
        results = {idx: {
            'total_score': 0,
            'total_steps': 0,
            'max_score': 0,
            'wins': 0,
            'total_position': 0,
            'games_played': 0
        } for idx in individual_indices}

        for game_num in range(num_games):
            # Create PvP game
            game = game_class(grid_size=20, num_snakes=num_snakes, display=False)

            # Set up all snakes with the individuals from the population
            networks = {}
            for i, idx in enumerate(individual_indices):
                networks[i] = game_individuals[i].network

            game.set_snake_networks(networks)
            states = game.reset()

            # Play the game
            while not game.done:
                actions = {}
                for snake_id, state in states.items():
                    if snake_id in networks:
                        actions[snake_id] = networks[snake_id].act(state)

                states, rewards, done, info = game.step(actions)

            # Calculate results for ALL individuals in this game (coevolution)
            snake_scores = game.get_snake_scores()
            winner = info.get('winner')

            # Update results for all individuals
            sorted_scores = sorted(snake_scores.values(), reverse=True)
            for i, idx in enumerate(individual_indices):
                score = snake_scores.get(i, 0)
                steps = game.steps
                won = (winner == i)

                results[idx]['total_score'] += score
                results[idx]['total_steps'] += steps
                results[idx]['max_score'] = max(results[idx]['max_score'], score)
                if won:
                    results[idx]['wins'] += 1

                # Calculate position (1st, 2nd, etc.)
                position = sorted_scores.index(score) + 1
                results[idx]['total_position'] += position
                results[idx]['games_played'] += 1

        # Calculate and update fitness for all individuals
        fitness_updates = {}
        for idx in individual_indices:
            res = results[idx]
            if res['games_played'] > 0:
                avg_score = res['total_score'] / res['games_played']
                avg_steps = res['total_steps'] / res['games_played']
                win_rate = res['wins'] / res['games_played']
                avg_position = res['total_position'] / res['games_played']

                # Enhanced fitness function for PvP
                fitness = (avg_score ** 1.5 * 100 +           # Non-linear score scaling
                          res['max_score'] * 50 +             # Bonus for best single game
                          win_rate * 2000 +                    # High bonus for winning
                          (4 - avg_position) * 500 +           # Position bonus (1st=3, 2nd=2, etc.)
                          min(avg_steps, 300) * 0.05 +         # Small efficiency bonus
                          (avg_score > 10) * 300 +             # Bonus for breaking score 10
                          (avg_score > 20) * 800)              # Large bonus for breaking score 20

                fitness_updates[idx] = {
                    'fitness': fitness,
                    'avg_score': avg_score,
                    'avg_position': avg_position,
                    'total_score': res['total_score'],
                    'max_score': res['max_score'],
                    'total_steps': res['total_steps'],
                    'wins': res['wins'],
                    'games_played': res['games_played']
                }

        return fitness_updates

    def evaluate_population(self, game_class, num_games=3, num_snakes=2, verbose=False, progress_callback=None):
        # Evaluate the entire population using coevolution: all individuals play against each other
        # This is true coevolution where all individuals evolve together
        start_time = time.time()

        if verbose:
            print(f"Gen {self.generation}: Evaluating {self.population_size} individuals (coevolution)...")

        # Initialize fitness accumulators for all individuals
        fitness_accumulators = {i: {
            'fitness_sum': 0.0,
            'score_sum': 0.0,
            'position_sum': 0.0,
            'total_score': 0,
            'max_score': 0,
            'total_steps': 0,
            'wins': 0,
            'games_played': 0
        } for i in range(self.population_size)}

        # For coevolution with 2 snakes: pair up individuals
        # Each individual plays against multiple others to get diverse fitness evaluation
        if num_snakes == 2:
            # Create pairs: each individual plays against several others
            # Use round-robin style: each individual plays against num_games different opponents
            eval_args = []
            for i in range(self.population_size):
                # Select opponents for this individual
                opponents = []
                for _ in range(num_games):
                    # Select a different opponent each game (not self)
                    opponent_idx = random.randint(0, self.population_size - 1)
                    while opponent_idx == i:
                        opponent_idx = random.randint(0, self.population_size - 1)
                    opponents.append(opponent_idx)

                # Create game groups: one game per opponent
                for opponent_idx in opponents:
                    eval_args.append((game_class, [i, opponent_idx], 1, num_snakes))
        else:
            # For more than 2 snakes, group individuals
            eval_args = []
            for _ in range(self.population_size * num_games):
                # Randomly select num_snakes individuals for each game
                selected = random.sample(range(self.population_size), min(num_snakes, self.population_size))
                eval_args.append((game_class, selected, 1, num_snakes))

        completed = 0
        total_games = len(eval_args)

        # Use ThreadPoolExecutor for parallel evaluation
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            # Submit all tasks
            futures = {executor.submit(self.evaluate_game_worker, args): args
                      for args in eval_args}

            # Process completed tasks
            for future in as_completed(futures):
                fitness_updates = future.result()
                completed += 1

                # Accumulate fitness updates for all individuals in the game
                for idx, update in fitness_updates.items():
                    acc = fitness_accumulators[idx]
                    acc['fitness_sum'] += update['fitness']
                    acc['score_sum'] += update['avg_score']
                    acc['position_sum'] += update['avg_position']
                    acc['total_score'] += update['total_score']
                    acc['max_score'] = max(acc['max_score'], update['max_score'])
                    acc['total_steps'] += update['total_steps']
                    acc['wins'] += update['wins']
                    acc['games_played'] += update['games_played']

                if progress_callback:
                    progress_callback(completed, total_games)
                elif verbose and completed % max(1, total_games // 10) == 0:
                    with self.print_lock:
                        print(f"  Progress: {completed}/{total_games} games")

        # Update all individuals with accumulated fitness
        for i, acc in fitness_accumulators.items():
            if acc['games_played'] > 0:
                self.population[i].fitness = acc['fitness_sum'] / acc['games_played']
                self.population[i].games_played = acc['games_played']
                self.population[i].total_score = acc['total_score']
                self.population[i].max_score = acc['max_score']
                self.population[i].total_steps = acc['total_steps']
                self.population[i].wins = acc['wins']
                self.population[i].avg_position = acc['position_sum'] / acc['games_played']
            else:
                # If no games played, set minimal fitness
                self.population[i].fitness = 0.0
                self.population[i].games_played = 0

        # Sort population by fitness (descending)
        self.population.sort(key=lambda x: x.fitness, reverse=True)

        # Update statistics
        fitnesses = [ind.fitness for ind in self.population]
        scores = [ind.total_score / ind.games_played if ind.games_played > 0 else 0 for ind in self.population]

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

    def roulette_wheel_selection(self):
        # Select an individual using roulette wheel selection
        # Based on fitness values (higher fitness = higher probability)
        wheel = sum(ind.fitness for ind in self.population)
        if wheel <= 0:
            # If all fitnesses are non-positive, use random selection
            return random.choice(self.population)

        pick = random.uniform(0, wheel)
        current = 0
        for individual in self.population:
            current += individual.fitness
            if current > pick:
                return individual
        # Fallback to best individual if something goes wrong
        return max(self.population, key=lambda x: x.fitness)

    def plus_selection(self, parent_population, offspring_population):
        # (μ+λ) selection: combine parents and offspring, select best num_parents
        combined = parent_population + offspring_population
        combined.sort(key=lambda x: x.fitness, reverse=True)
        return combined[:self.num_parents]

    def sbx_crossover(self, parent1, parent2, eta=100.0):
        # Simulated Binary Crossover (SBX)
        # eta: distribution index (higher = more similar to parents)
        # Changed default eta from 2.0 to 100.0 to match SnakeAI approach
        weights1 = parent1.get_weights()
        weights2 = parent2.get_weights()

        u = np.random.random(len(weights1))
        beta = np.zeros(len(weights1))

        for i in range(len(weights1)):
            if u[i] <= 0.5:
                beta[i] = (2 * u[i]) ** (1.0 / (eta + 1))
            else:
                beta[i] = (1.0 / (2 * (1 - u[i]))) ** (1.0 / (eta + 1))

        offspring1_weights = 0.5 * ((1 + beta) * weights1 + (1 - beta) * weights2)
        offspring2_weights = 0.5 * ((1 - beta) * weights1 + (1 + beta) * weights2)

        offspring1 = Individual(input_size=17, hidden_size=64, output_size=3, device=self.device)
        offspring2 = Individual(input_size=17, hidden_size=64, output_size=3, device=self.device)
        offspring1.set_weights(offspring1_weights)
        offspring2.set_weights(offspring2_weights)

        return offspring1, offspring2

    def spbx_crossover(self, parent1, parent2):
        # Single-Point Binary Crossover (SPBX) - similar to single-point but for real-valued
        weights1 = parent1.get_weights()
        weights2 = parent2.get_weights()

        # Single-point crossover
        crossover_point = random.randint(1, len(weights1) - 1)

        offspring1_weights = np.concatenate([weights1[:crossover_point], weights2[crossover_point:]])
        offspring2_weights = np.concatenate([weights2[:crossover_point], weights1[crossover_point:]])

        offspring1 = Individual(input_size=17, hidden_size=64, output_size=3, device=self.device)
        offspring2 = Individual(input_size=17, hidden_size=64, output_size=3, device=self.device)
        offspring1.set_weights(offspring1_weights)
        offspring2.set_weights(offspring2_weights)

        return offspring1, offspring2

    def crossover(self, parent1, parent2):
        # Create two offspring through crossover
        # 50% SBX, 50% SPBX
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()

        # Choose crossover type: 50% SBX, 50% SPBX
        if self.crossover_type == 'mixed':
            if random.random() < 0.5:
                return self.sbx_crossover(parent1, parent2)
            else:
                return self.spbx_crossover(parent1, parent2)
        elif self.crossover_type == 'sbx':
            return self.sbx_crossover(parent1, parent2)
        else:  # spbx
            return self.spbx_crossover(parent1, parent2)

    def mutate(self, individual):
        # Mutation: 100% Gaussian, 5% static rate
        weights = individual.get_weights()

        # Static mutation strength (not adaptive)
        mutation_strength = 0.1

        # Apply Gaussian mutation (100%)
        for i in range(len(weights)):
            if random.random() < self.mutation_rate:  # 5% mutation rate
                weights[i] += np.random.normal(0, mutation_strength)

        individual.set_weights(weights)

    def evolve_generation(self, verbose=False, parent_selection='roulette_wheel'):
        # Create the next generation using genetic operations
        # Selection type: 'plus' uses (μ+λ) selection, otherwise tournament
        # parent_selection: 'roulette_wheel' or 'tournament' for selecting parents
        parent_population = [ind.copy() for ind in self.population]
        offspring_population = []

        # Generate offspring (num_offspring individuals)
        while len(offspring_population) < self.num_offspring:
            # Select parents using roulette wheel (matching SnakeAI approach)
            if parent_selection == 'roulette_wheel':
                parent1 = self.roulette_wheel_selection()
                parent2 = self.roulette_wheel_selection()
            else:
                parent1 = self.tournament_selection()
                parent2 = self.tournament_selection()

            # Create offspring
            offspring1, offspring2 = self.crossover(parent1, parent2)

            # Mutate offspring
            self.mutate(offspring1)
            self.mutate(offspring2)

            # Add to offspring population
            offspring_population.extend([offspring1, offspring2])

        # Trim to exact num_offspring
        offspring_population = offspring_population[:self.num_offspring]

        # Apply selection: 'plus' uses (μ+λ), otherwise use offspring only
        if self.selection_type == 'plus':
            # (μ+λ) selection: combine parents and offspring, select best num_parents
            self.population = self.plus_selection(parent_population, offspring_population)
        else:
            # Standard selection: use offspring only, select best num_parents
            offspring_population.sort(key=lambda x: x.fitness, reverse=True)
            self.population = offspring_population[:self.num_parents]

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
        individual = Individual(input_size=17, hidden_size=64, output_size=3, device=self.device)
        individual.network.load_state_dict(torch.load(filepath, map_location=self.device, weights_only=False))
        return individual

    def save_state(self, filepath):
        # Save the complete GeneticAlgorithm state for resuming training
        # This saves the entire population, generation number, and history
        from pathlib import Path
        state = {
            'generation': self.generation,
            'population': [ind.get_weights() for ind in self.population],
            'best_fitness_history': self.best_fitness_history,
            'avg_fitness_history': self.avg_fitness_history,
            'best_score_history': self.best_score_history,
            'avg_score_history': self.avg_score_history,
            'num_parents': self.num_parents,
            'num_offspring': self.num_offspring,
            'mutation_rate': self.mutation_rate,
            'crossover_rate': self.crossover_rate,
            'elitism_ratio': self.elitism_ratio,
            'tournament_size': self.tournament_size,
            'selection_type': self.selection_type,
            'crossover_type': self.crossover_type,
        }
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, filepath)

    def load_state(self, filepath):
        # Load the complete GeneticAlgorithm state from a saved checkpoint
        state = torch.load(filepath, map_location=self.device, weights_only=False)

        self.generation = state['generation']
        self.best_fitness_history = state['best_fitness_history']
        self.avg_fitness_history = state['avg_fitness_history']
        self.best_score_history = state['best_score_history']
        self.avg_score_history = state['avg_score_history']

        # Reconstruct population from saved weights
        self.population = []
        for weights in state['population']:
            individual = Individual(input_size=17, hidden_size=64, output_size=3, device=self.device)
            individual.set_weights(weights)
            self.population.append(individual)

        return self
