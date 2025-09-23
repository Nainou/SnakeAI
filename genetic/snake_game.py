import torch
import torch.nn as nn
import numpy as np
import random
import copy
import matplotlib.pyplot as plt
from collections import deque
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

class NeuralNetwork(nn.Module):
    def __init__(self, input_size=17, hidden_size=32, output_size=3, device=None):
        """
        Simple neural network for genetic algorithm
        Smaller than DQN to make evolution more manageable
        """
        super(NeuralNetwork, self).__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)

        # Move to device
        self.to(self.device)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Individual:
    def __init__(self, input_size=17, hidden_size=32, output_size=3, device=None):
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
    def __init__(self, population_size=50, mutation_rate=0.1, crossover_rate=0.7,
                 elitism_ratio=0.1, tournament_size=3, num_threads=4, device=None):
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

        # Fitness function: prioritize score, but also reward efficiency and wins
        fitness = (avg_score * 100 +  # High weight on score
                  max_score * 50 +    # Bonus for best single game
                  win_rate * 1000 +   # Huge bonus for winning games
                  min(avg_steps, 200) * 0.1)  # Small bonus for efficiency (capped)

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
        """Mutate an individual's weights"""
        weights = individual.get_weights()

        # Add Gaussian noise to weights with given probability
        for i in range(len(weights)):
            if random.random() < self.mutation_rate:
                weights[i] += np.random.normal(0, 0.1)  # Small Gaussian noise

        individual.set_weights(weights)

    def evolve_generation(self):
        """Create the next generation using genetic operations"""
        new_population = []

        # Elitism: keep the best individuals
        elite_count = int(self.population_size * self.elitism_ratio)
        for i in range(elite_count):
            new_population.append(self.population[i].copy())

        # Generate offspring to fill the rest of the population
        while len(new_population) < self.population_size:
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
    from snake_game import SnakeGameRL

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
    from snake_game import SnakeGameRL

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
                import pygame
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        game.close()
                        return scores

        scores.append(game.score)
        print(f"Game {i+1}: Score = {game.score}")

    if display:
        game.close()

    print(f"\nTest Results:")
    print(f"Average Score: {np.mean(scores):.2f}")
    print(f"Max Score: {max(scores)}")
    print(f"Min Score: {min(scores)}")
    print(f"Standard Deviation: {np.std(scores):.2f}")

    return scores

if __name__ == "__main__":
    print("Genetic Algorithm Snake AI")
    print("=" * 50)

    # Train the genetic algorithm
    ga = train_genetic_algorithm(generations=30, population_size=30, games_per_eval=3, verbose=True)

    # Plot the evolution progress
    print("\nPlotting evolution progress...")
    plot_evolution_progress(ga)

    # Test the best individual
    print("\nTesting best individual...")
    test_scores = test_genetic_individual('genetic_snake_final.pth', num_games=5, display=True)
