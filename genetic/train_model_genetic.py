from snake_game_genetic import train_genetic_algorithm, plot_training_progress, plot_epoch_max_scores, print_training_statistics
import sys
import multiprocessing
from pathlib import Path
import re

def get_optimal_threads(population_size=200, device=None):
    cpu_count = multiprocessing.cpu_count()
    # Scale threads with population size, but cap reasonably
    # For larger populations, use more threads
    # For CUDA, use fewer threads to avoid GPU contention
    import torch
    is_cuda = device is not None and device.type == 'cuda' if device else torch.cuda.is_available()

    if is_cuda:
        # CUDA: Use fewer threads to avoid GPU contention and potential deadlocks
        if population_size >= 500:
            return min(8, cpu_count)  # Max 8 threads for CUDA
        elif population_size >= 200:
            return min(6, cpu_count)
        else:
            return min(4, cpu_count)
    else:
        # CPU: Can use more threads
        if population_size >= 500:
            return min(cpu_count, 16)  # Up to 16 threads for large populations
        elif population_size >= 200:
            return min(cpu_count - 1, 12)  # Up to 12 threads for medium populations
        else:
            return min(max(2, cpu_count - 1), 8)  # Default: leave 1 core free, max 8 threads

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

    # Use SnakeAI approach: 500 parents, 1000 offspring
    num_parents = 500
    num_offspring = 1000

    # Determine device first to calculate optimal threads
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    threads = get_optimal_threads(num_parents, device)
    print(f"Parents: {num_parents}, Offspring: {num_offspring}")
    print(f"Threads: {threads} (optimized for {device.type.upper()})")
    print()

    # Check for existing checkpoints to resume from
    saved_dir = Path('saved')
    resume_from = None

    # Look for full state files first (best option)
    state_files = list(saved_dir.glob('genetic_*.state.pth'))
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
        # Look for model files as fallback (matches pattern: genetic_32_20_12_4_gen520.pth)
        model_files = list(saved_dir.glob('genetic_*gen*.pth'))
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
    # resume_from = 'saved/genetic_32_20_12_4_gen520.pth'
    # Or set to None to start fresh:
    # resume_from = None

    # Train the genetic algorithm
    ga = train_genetic_algorithm(
        generations=1500,
        population_size=num_parents,  # Used as fallback if num_parents not set
        games_per_eval=3,
        num_threads=threads,
        quiet=True,
        verbose=True,
        device=device,
        num_parents=num_parents,
        num_offspring=num_offspring,
        resume_from=resume_from
    )

    # Get the actual saved filename
    saved_dir = Path('saved')
    final_models = list(saved_dir.glob('genetic_*_final.pth'))
    if final_models:
        print(f"\nModel saved as '{final_models[0].name}'")
    else:
        print("\nModel saved (check saved/ directory)")

    # Print training statistics
    print_training_statistics(ga)

    # Plot training progress (moving average)
    plot_training_progress(ga)

    # Plot max scores per generation
    plot_epoch_max_scores(ga)
