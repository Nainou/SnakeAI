"""
Headless Training Script for PvP Snake AI

Trains multiple snakes using genetic algorithm to compete against each other.
Runs without rendering for maximum speed.
"""

import sys
import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from game.snake_game_pvp import SnakeGamePvP
from ai.genetic_algorithm import GeneticAlgorithm


def train_pvp_genetic_algorithm(generations=50, population_size=50, games_per_eval=3,
                               num_snakes=2, verbose=True, num_threads=4, quiet=False, device=None):
    """
    Train a genetic algorithm for PvP Snake gameplay

    Args:
        generations: Number of generations to evolve
        population_size: Size of population in each generation
        games_per_eval: Number of games to play for fitness evaluation
        num_snakes: Number of snakes in each PvP game (2-4)
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
        print(f"🧬 PvP Genetic Algorithm Training")
        print(f"Population: {population_size}, Generations: {generations}, Threads: {num_threads}")
        print(f"Snakes per game: {num_snakes}, Device: {device}")
        print("=" * 60)
    else:
        print(f"🧬 PvP Training on {device} | Pop: {population_size} | Gen: {generations} | Snakes: {num_snakes}")

    start_time = time.time()

    for gen in range(generations):
        gen_start = time.time()

        # Evaluate current generation
        if quiet:
            # Simple progress callback for quiet mode
            def progress_callback(completed, total):
                if completed % max(1, total // 4) == 0:
                    print(f"Gen {gen}: {completed}/{total}", end='\r')

            ga.evaluate_population(SnakeGamePvP, num_games=games_per_eval, num_snakes=num_snakes,
                                 verbose=False, progress_callback=progress_callback)

            # Show generation results in quiet mode
            best_score = ga.best_score_history[-1]
            avg_score = ga.avg_score_history[-1]
            best_position = ga.population[0].avg_position
            gen_time = time.time() - gen_start
            print(f"Gen {gen:2d}: Best={best_score:4.1f} Avg={avg_score:4.1f} Pos={best_position:.1f} ({gen_time:4.1f}s)")
        else:
            ga.evaluate_population(SnakeGamePvP, num_games=games_per_eval, num_snakes=num_snakes, verbose=verbose)

        # Save best individual periodically
        if gen % 10 == 0:
            os.makedirs('saved', exist_ok=True)
            ga.save_best(f'pvp/saved/pvp_snake_gen_{gen}.pth')

        # Evolve to next generation (except for the last generation)
        if gen < generations - 1:
            ga.evolve_generation()

        # Show generation summary
        gen_time = time.time() - gen_start
        if not quiet and gen % 5 == 0:
            best_score = ga.best_score_history[-1]
            avg_score = ga.avg_score_history[-1]
            best_position = ga.population[0].avg_position
            print(f"Generation {gen} complete: Best={best_score:.1f}, Avg={avg_score:.1f}, Pos={best_position:.1f} ({gen_time:.1f}s)")

    # Save final best individual
    os.makedirs('saved', exist_ok=True)
    ga.save_best('pvp/saved/pvp_snake_final.pth')

    elapsed_time = time.time() - start_time

    if not quiet:
        print(f"\n✅ PvP Training completed in {elapsed_time:.1f} seconds")
        print(f"Final best score: {ga.best_score_history[-1]:.1f}")
        print(f"Final best position: {ga.population[0].avg_position:.1f}")
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

    # Plot position distribution for last generation
    plt.subplot(2, 2, 4)
    last_gen_positions = [ind.avg_position for ind in ga.population]
    plt.hist(last_gen_positions, bins=20, alpha=0.7, color='cyan')
    plt.title(f'Position Distribution - Generation {ga.generation-1}')
    plt.xlabel('Average Position')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main():
    """Main training function with configuration options"""
    print("🐍 PvP Snake AI Training")
    print("=" * 50)
    print("Choose training mode:")
    print("1. Quick train (20 gens, 30 pop, 2 snakes)")
    print("2. Medium train (40 gens, 50 pop, 2 snakes)")
    print("3. Full train (60 gens, 80 pop, 4 snakes)")
    print("4. Custom configuration")

    choice = input("\nChoice (1-4): ").strip()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    threads = min(max(2, torch.get_num_threads() - 1), 8)

    if choice == "1":
        ga = train_pvp_genetic_algorithm(
            generations=20, population_size=30, games_per_eval=3,
            num_snakes=2, num_threads=threads, quiet=True, device=device
        )
    elif choice == "2":
        ga = train_pvp_genetic_algorithm(
            generations=40, population_size=50, games_per_eval=3,
            num_snakes=2, num_threads=threads, quiet=True, device=device
        )
    elif choice == "3":
        ga = train_pvp_genetic_algorithm(
            generations=60, population_size=80, games_per_eval=3,
            num_snakes=4, num_threads=threads, quiet=True, device=device
        )
    elif choice == "4":
        try:
            generations = int(input("Generations (default 40): ") or "40")
            population_size = int(input("Population size (default 50): ") or "50")
            games_per_eval = int(input("Games per evaluation (default 3): ") or "3")
            num_snakes = int(input("Snakes per game (2-4, default 2): ") or "2")
            num_snakes = max(2, min(4, num_snakes))
            threads = int(input(f"Threads (default {threads}): ") or str(threads))
            quiet = input("Quiet mode? (Y/n): ").strip().lower() != 'n'
        except ValueError:
            print("Invalid input, using defaults...")
            generations, population_size, games_per_eval, num_snakes = 40, 50, 3, 2
            quiet = True

        ga = train_pvp_genetic_algorithm(
            generations=generations, population_size=population_size,
            games_per_eval=games_per_eval, num_snakes=num_snakes,
            num_threads=threads, quiet=quiet, device=device
        )
    else:
        print("Invalid choice, using quick train...")
        ga = train_pvp_genetic_algorithm(
            generations=20, population_size=30, games_per_eval=3,
            num_snakes=2, num_threads=threads, quiet=True, device=device
        )

    if ga is None:
        return

    # Post-training options
    print(f"\n📊 Results: Best={ga.best_score_history[-1]:.1f}, "
          f"Position={ga.population[0].avg_position:.1f}, "
          f"Improvement={ga.best_score_history[-1] - ga.best_score_history[0]:.1f}")

    # Plot evolution progress
    plot_choice = input("Show evolution plots? (y/N): ").strip().lower()
    if plot_choice == 'y':
        plot_evolution_progress(ga)

    print("\n✅ Training complete! Use 'python test/test_pvp.py' to test the models.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)
