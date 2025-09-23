#!/usr/bin/env python3
"""
Training script for the genetic algorithm Snake AI with threading support
"""

from genetic_snake import train_genetic_algorithm, plot_evolution_progress, test_genetic_individual
import sys
import os
import multiprocessing

def get_optimal_threads():
    """Get optimal number of threads based on CPU cores"""
    cpu_count = multiprocessing.cpu_count()
    return min(max(2, cpu_count - 1), 8)  # Leave 1 core free, max 8 threads

def quick_train():
    """Quick training with sensible defaults"""
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    threads = get_optimal_threads()

    print(f"🚀 Quick Training Mode")
    print(f"Device: {device} | Threads: {threads}")

    ga = train_genetic_algorithm(
        generations=15,
        population_size=20,
        games_per_eval=3,
        num_threads=threads,
        quiet=True,
        verbose=False,
        device=device
    )

    print(f"✅ Quick training complete! Best score: {ga.best_score_history[-1]:.1f}")
    return ga

def custom_train():
    """Custom training with user preferences"""
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("⚙️ Custom Training Configuration")
    print(f"Auto-detected device: {device}")

    try:
        generations = int(input("Generations (default 20): ") or "20")
        population_size = int(input("Population size (default 25): ") or "25")
        games_per_eval = int(input("Games per evaluation (default 3): ") or "3")
        threads = int(input(f"Threads (default {get_optimal_threads()}): ") or str(get_optimal_threads()))
        quiet = input("Quiet mode? (Y/n): ").strip().lower() != 'n'
    except ValueError:
        print("Invalid input, using defaults...")
        generations, population_size, games_per_eval = 20, 25, 3
        threads = get_optimal_threads()
        quiet = True

    total_evals = generations * population_size * games_per_eval
    speed_multiplier = 10 if device.type == 'cuda' else 3  # GPU is ~3x faster than estimation

    print(f"\n📋 Configuration:")
    print(f"  Device: {device}")
    print(f"  Generations: {generations}")
    print(f"  Population: {population_size}")
    print(f"  Games/eval: {games_per_eval}")
    print(f"  Threads: {threads}")
    print(f"  Total evaluations: {total_evals}")
    print(f"  Estimated time: ~{total_evals / (threads * speed_multiplier):.1f} minutes")

    confirm = input("\nStart training? (y/N): ").strip().lower()
    if confirm != 'y':
        print("Training cancelled.")
        return None

    ga = train_genetic_algorithm(
        generations=generations,
        population_size=population_size,
        games_per_eval=games_per_eval,
        num_threads=threads,
        quiet=quiet,
        verbose=not quiet,
        device=device
    )

    print(f"✅ Training complete! Best score: {ga.best_score_history[-1]:.1f}")
    return ga

def main():
    print("🐍 Genetic Algorithm Snake AI Training")
    print("=" * 50)
    print("Choose training mode:")
    print("1. Quick train (15 gens, 20 pop, fast)")
    print("2. Custom configuration")
    print("3. Test existing model")

    choice = input("\nChoice (1-3): ").strip()

    ga = None
    if choice == "1":
        ga = quick_train()
    elif choice == "2":
        ga = custom_train()
    elif choice == "3":
        if os.path.exists('genetic_snake_final.pth'):
            print("🎮 Testing existing model...")
            test_genetic_individual('genetic_snake_final.pth', num_games=5, display=True)
        else:
            print("❌ No trained model found. Train one first!")
        return
    else:
        print("Invalid choice, using quick train...")
        ga = quick_train()

    if ga is None:
        return

    # Post-training options
    print(f"\n📊 Results: Best={ga.best_score_history[-1]:.1f}, "
          f"Improvement={ga.best_score_history[-1] - ga.best_score_history[0]:.1f}")

    # Plot evolution progress
    plot_choice = input("Show evolution plots? (y/N): ").strip().lower()
    if plot_choice == 'y':
        plot_evolution_progress(ga)

    # Test the best individual
    test_choice = input("Test best individual? (y/N): ").strip().lower()
    if test_choice == 'y':
        print("🎮 Testing best individual...")
        test_genetic_individual('genetic_snake_final.pth', num_games=5, display=True)

    print("\n✅ All done! Use 'python test.py' to test anytime.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)
