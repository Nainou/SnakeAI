from snake_game_genetic import train_genetic_algorithm, plot_training_progress, plot_epoch_max_scores, print_training_statistics
import sys
import multiprocessing
from pathlib import Path

def get_optimal_threads():
    cpu_count = multiprocessing.cpu_count()
    return min(max(2, cpu_count - 1), 8)  # Leave 1 core free, max 8 threads

if __name__ == "__main__":
    # Example usage
    print("Training Genetic Algorithm for Snake Game...")
    print("=" * 50)

    import torch

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

    threads = get_optimal_threads()
    print(f"Threads: {threads}")
    print()

    # Train the genetic algorithm
    ga = train_genetic_algorithm(
        generations=1500,
        population_size=200,
        games_per_eval=3,
        num_threads=threads,
        quiet=True,
        verbose=True,
        device=device
    )

    # Get the actual saved filename
    saved_dir = Path('genetic/saved')
    final_models = list(saved_dir.glob('genetic_*_final.pth'))
    if final_models:
        print(f"\nModel saved as '{final_models[0].name}'")
    else:
        print("\nModel saved (check genetic/saved/ directory)")

    # Print training statistics
    print_training_statistics(ga)

    # Plot training progress (moving average)
    plot_training_progress(ga)

    # Plot max scores per generation
    plot_epoch_max_scores(ga)
