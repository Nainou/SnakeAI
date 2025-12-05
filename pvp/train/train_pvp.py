# Headless Training Script for PvP Snake AI
# Trains multiple snakes using genetic algorithm to compete against each other.
# Runs without rendering for maximum speed.

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


def resume_from_checkpoint(checkpoint_path, generations=50, population_size=50, games_per_eval=3,
                          num_snakes=2, verbose=True, num_threads=4, quiet=False, device=None,
                          num_parents=None, num_offspring=None):
    # Resume training from a checkpoint file
    # If checkpoint is a full state file (.state.pth), loads complete state
    # If checkpoint is a model file (.pth), loads best individual and seeds population
    # Args:
    #   checkpoint_path: Path to checkpoint file (full state or model file)
    #   Other args same as train_pvp_genetic_algorithm
    import re

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = Path(checkpoint_path)

    # Check if it's a full state file
    if checkpoint_path.suffix == '.pth' and 'state' in checkpoint_path.stem.lower():
        # Load full state
        ga = GeneticAlgorithm(
            population_size=population_size,
            num_threads=num_threads,
            device=device,
            mutation_rate=0.05,
            selection_type='plus',
            crossover_type='mixed',
            num_parents=num_parents,
            num_offspring=num_offspring
        )
        ga.load_state(checkpoint_path)
        start_gen = ga.generation
        if not quiet:
            print(f"Resuming from generation {start_gen} (full state loaded)")
    else:
        # Load best individual from model file and seed population
        ga = GeneticAlgorithm(
            population_size=population_size,
            num_threads=num_threads,
            device=device,
            mutation_rate=0.05,
            selection_type='plus',
            crossover_type='mixed',
            num_parents=num_parents,
            num_offspring=num_offspring
        )

        # Extract generation number from filename
        gen_match = re.search(r'gen[_\s]*(\d+)', checkpoint_path.stem.lower())
        if gen_match:
            start_gen = int(gen_match.group(1))
        else:
            start_gen = 0
            if not quiet:
                print("Warning: Could not extract generation number from filename, starting from 0")

        # Load the best individual
        best_individual = ga.load_individual(checkpoint_path)

        # Seed population: first individual is the loaded one, rest are mutations
        ga.population[0] = best_individual
        for i in range(1, len(ga.population)):
            # Create a mutated copy
            mutated = best_individual.copy()
            ga.mutate(mutated)
            ga.population[i] = mutated

        if not quiet:
            print(f"Resuming from generation {start_gen} (loaded best individual and seeded population)")

    # Continue training from start_gen
    if not quiet:
        print(f"Continuing training for {generations - start_gen} more generations")
        print(f"Population: {ga.population_size}, Threads: {num_threads}")
        print(f"Device: {device}")
        print("=" * 60)
    else:
        print(f"Resuming training on {device} | Pop: {ga.population_size} | Gen: {start_gen}-{generations} | Threads: {num_threads}")

    start_time = time.time()
    generation_times = []

    for gen in range(start_gen, generations):
        gen_start = time.time()

        # Evaluate current generation
        if quiet:
            def progress_callback(completed, total):
                if completed % max(1, total // 4) == 0:
                    print(f"Gen {gen}: {completed}/{total}", end='\r')

            ga.evaluate_population(SnakeGamePvP, num_games=games_per_eval, num_snakes=num_snakes,
                                 verbose=False, progress_callback=progress_callback)

            best_score = ga.best_score_history[-1]
            avg_score = ga.avg_score_history[-1]
            best_position = ga.population[0].avg_position
            gen_time = time.time() - gen_start
            print(f"Gen {gen:2d}: Best={best_score:4.1f} Avg={avg_score:4.1f} Pos={best_position:.1f} ({gen_time:4.1f}s)")
        else:
            ga.evaluate_population(SnakeGamePvP, num_games=games_per_eval, num_snakes=num_snakes, verbose=verbose)

        # Save best individual periodically
        if gen % 10 == 0:
            os.makedirs('pvp/saved', exist_ok=True)
            ga.save_best(f'pvp/saved/pvp_snake_gen_{gen}.pth')
            # Also save full state for better resume capability
            ga.save_state(f'pvp/saved/pvp_snake_gen_{gen}.state.pth')

        # Evolve to next generation (except for the last generation)
        if gen < generations - 1:
            ga.evolve_generation(verbose=verbose, parent_selection='roulette_wheel')

        # Show generation summary
        gen_time = time.time() - gen_start
        generation_times.append(gen_time)
        if not quiet and gen % 5 == 0:
            best_score = ga.best_score_history[-1]
            avg_score = ga.avg_score_history[-1]
            best_position = ga.population[0].avg_position
            print(f"Generation {gen} complete: Best={best_score:.1f}, Avg={avg_score:.1f}, Pos={best_position:.1f} ({gen_time:.1f}s)")

    # Save final best individual
    os.makedirs('pvp/saved', exist_ok=True)
    ga.save_best('pvp/saved/pvp_snake_final.pth')
    ga.save_state('pvp/saved/pvp_snake_final.state.pth')

    elapsed_time = time.time() - start_time

    if not quiet:
        print(f"\nTraining completed in {elapsed_time:.1f} seconds")
        print(f"Final best score: {ga.best_score_history[-1]:.1f}")
        print(f"Final best position: {ga.population[0].avg_position:.1f}")
        print(f"Improvement: {ga.best_score_history[-1] - ga.best_score_history[0]:.1f} points")

    return ga


def train_pvp_genetic_algorithm(generations=50, population_size=50, games_per_eval=3,
                               num_snakes=2, verbose=True, num_threads=4, quiet=False, device=None,
                               num_parents=None, num_offspring=None, resume_from=None):
    # Train a genetic algorithm for PvP Snake gameplay
    # Args:
    #   generations: Number of generations to evolve
    #   population_size: Size of population in each generation (used if num_parents/num_offspring not set)
    #   games_per_eval: Number of games to play for fitness evaluation
    #   num_snakes: Number of snakes in each PvP game (2-4)
    #   verbose: Print detailed progress
    #   num_threads: Number of threads for parallel evaluation
    #   quiet: Minimal output mode
    #   device: Device to run on (cuda/cpu)
    #   num_parents: Number of parents (if None, uses population_size)
    #   num_offspring: Number of offspring to generate (if None, uses population_size)
    #   resume_from: Path to checkpoint file to resume from (if None, starts fresh)

    # If resume_from is provided, use resume function instead
    if resume_from is not None:
        return resume_from_checkpoint(
            resume_from, generations, population_size, games_per_eval,
            num_snakes, verbose, num_threads, quiet, device, num_parents, num_offspring
        )

    # Device setup
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize genetic algorithm with settings matching genetic training
    ga = GeneticAlgorithm(
        population_size=population_size,
        num_threads=num_threads,
        device=device,
        mutation_rate=0.05,  # Static 5% mutation rate
        selection_type='plus',  # (μ+λ) selection
        crossover_type='mixed',  # 50% SBX, 50% SPBX
        num_parents=num_parents,
        num_offspring=num_offspring
    )

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
            os.makedirs('pvp/saved', exist_ok=True)
            ga.save_best(f'pvp/saved/pvp_snake_gen_{gen}.pth')
            # Also save full state for better resume capability
            ga.save_state(f'pvp/saved/pvp_snake_gen_{gen}.state.pth')

        # Evolve to next generation (except for the last generation)
        if gen < generations - 1:
            ga.evolve_generation(verbose=verbose, parent_selection='roulette_wheel')

        # Show generation summary
        gen_time = time.time() - gen_start
        if not quiet and gen % 5 == 0:
            best_score = ga.best_score_history[-1]
            avg_score = ga.avg_score_history[-1]
            best_position = ga.population[0].avg_position
            print(f"Generation {gen} complete: Best={best_score:.1f}, Avg={avg_score:.1f}, Pos={best_position:.1f} ({gen_time:.1f}s)")

    # Save final best individual
    os.makedirs('pvp/saved', exist_ok=True)
    ga.save_best('pvp/saved/pvp_snake_final.pth')
    ga.save_state('pvp/saved/pvp_snake_final.state.pth')

    elapsed_time = time.time() - start_time

    if not quiet:
        print(f"\nPvP Training completed in {elapsed_time:.1f} seconds")
        print(f"Final best score: {ga.best_score_history[-1]:.1f}")
        print(f"Final best position: {ga.population[0].avg_position:.1f}")
        print(f"Improvement: {ga.best_score_history[-1] - ga.best_score_history[0]:.1f} points")

    return ga


def plot_evolution_progress(ga):
    # Plot the evolution progress over generations
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
    # Main training function matching genetic training structure
    import multiprocessing
    from pathlib import Path
    import re

    def get_optimal_threads(population_size=200):
        cpu_count = multiprocessing.cpu_count()
        # Scale threads with population size, but cap reasonably
        # For larger populations, use more threads
        if population_size >= 500:
            return min(cpu_count, 16)  # Up to 16 threads for large populations
        elif population_size >= 200:
            return min(cpu_count - 1, 12)  # Up to 12 threads for medium populations
        else:
            return min(max(2, cpu_count - 1), 8)  # Default: leave 1 core free, max 8 threads

    print("Training Genetic Algorithm for PvP Snake Game...")
    print("=" * 50)

    # Check CUDA availability
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ CUDA available - Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device("cpu")
        print("  WARNING: CUDA not available - Running on CPU!")
        print("   Training will be MUCH slower. Install PyTorch with CUDA support for GPU acceleration.")
        print("   Check: python -c 'import torch; print(torch.cuda.is_available())'")

    # Use SnakeAI approach: 500 parents, 1000 offspring
    num_parents = 500
    num_offspring = 1000
    threads = get_optimal_threads(num_parents)
    print(f"Parents: {num_parents}, Offspring: {num_offspring}")
    print(f"Threads: {threads}")
    print()

    # Check for existing checkpoints to resume from
    saved_dir = Path('pvp/saved')
    resume_from = None

    # Look for full state files first (best option)
    state_files = list(saved_dir.glob('pvp_*.state.pth'))
    if state_files:
        # Sort by generation number (extract from filename)
        def get_gen_number(path):
            match = re.search(r'gen[_\s]*(\d+)', path.stem.lower())
            return int(match.group(1)) if match else -1

        state_files.sort(key=get_gen_number, reverse=True)
        resume_from = str(state_files[0])
        gen_num = get_gen_number(state_files[0])
        print(f"Found checkpoint: {resume_from} (generation {gen_num})")
        print("Resuming training from checkpoint...")
        print()
    else:
        # Look for model files as fallback
        model_files = list(saved_dir.glob('pvp_*gen*.pth'))
        if model_files:
            def get_gen_number(path):
                match = re.search(r'gen[_\s]*(\d+)', path.stem.lower())
                return int(match.group(1)) if match else -1

            model_files.sort(key=get_gen_number, reverse=True)
            resume_from = str(model_files[0])
            gen_num = get_gen_number(model_files[0])
            print(f"Found model checkpoint: {resume_from} (generation {gen_num})")
            print("Resuming training from model checkpoint (will seed population)...")
            print()

    # You can also manually specify a checkpoint:
    # resume_from = 'pvp/saved/pvp_snake_gen_20.pth'
    # Or set to None to start fresh:
    # resume_from = None

    # Train the genetic algorithm
    ga = train_pvp_genetic_algorithm(
        generations=1500,
        population_size=num_parents,  # Used as fallback if num_parents not set
        games_per_eval=3,
        num_snakes=2,  # Keep 2 snakes for PvP
        num_threads=threads,
        quiet=True,
        verbose=True,
        device=device,
        num_parents=num_parents,
        num_offspring=num_offspring,
        resume_from=resume_from
    )

    if ga is None:
        return

    # Get the actual saved filename
    saved_dir = Path('pvp/saved')
    final_models = list(saved_dir.glob('pvp_*_final.pth'))
    if final_models:
        print(f"\nModel saved as '{final_models[0].name}'")
    else:
        print("\nModel saved (check saved/ directory)")

    # Post-training results
    print(f"\nResults: Best={ga.best_score_history[-1]:.1f}, "
          f"Position={ga.population[0].avg_position:.1f}, "
          f"Improvement={ga.best_score_history[-1] - ga.best_score_history[0]:.1f}")

    # Plot evolution progress
    plot_choice = input("\nShow evolution plots? (y/N): ").strip().lower()
    if plot_choice == 'y':
        plot_evolution_progress(ga)

    print("\nTraining complete! Use 'python test/test_pvp.py' to test the models.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)
