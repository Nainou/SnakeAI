import matplotlib.pyplot as plt
import re
import os

def parse_genetic_log(filepath):
    """Parse the genetic generation log file"""
    generations = []
    best_scores = []
    avg_scores = []
    times = []

    # Track seen generations to handle duplicates (keep last occurrence)
    seen_generations = {}

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Match pattern: Gen  X: Best= Y.Y Avg= Z.Z (TIME)
            # Handle both formats: "Avg= 0.0" and "Avg=0.0" (with or without space)
            # Also extract time: ( 3.0s) or (408.6s)
            match = re.match(r'Gen\s+(\d+):\s+Best=\s*([\d.]+)\s+Avg=\s*([\d.]+)\s+\([\s]*([\d.]+)s\)', line)
            if match:
                gen = int(match.group(1))
                best = float(match.group(2))
                avg = float(match.group(3))
                time = float(match.group(4))

                # Store or update if we've seen this generation before
                seen_generations[gen] = (best, avg, time)

    # Convert to sorted lists - fill in any missing generations with previous values
    if not seen_generations:
        return [], [], [], []

    min_gen = min(seen_generations.keys())
    max_gen = max(seen_generations.keys())

    # Fill in all generations from min to max
    last_best = None
    last_avg = None
    last_time = None
    for gen in range(min_gen, max_gen + 1):
        if gen in seen_generations:
            last_best = seen_generations[gen][0]
            last_avg = seen_generations[gen][1]
            last_time = seen_generations[gen][2]

        if last_best is not None and last_avg is not None and last_time is not None:
            generations.append(gen)
            best_scores.append(last_best)
            avg_scores.append(last_avg)
            times.append(last_time)

    return generations, best_scores, avg_scores, times

def plot_genetic_training(filepath='genetic_gen.txt', window=10):
    """Plot genetic algorithm training progress"""
    # Parse the log file
    generations, best_scores, avg_scores, times = parse_genetic_log(filepath)

    if not generations:
        print(f"No data found in {filepath}")
        return

    # Plot 1: Average Score
    plt.figure(figsize=(10, 6))

    # Calculate moving average
    if len(avg_scores) >= window:
        moving_avg = [sum(avg_scores[i:i+window]) / window
                     for i in range(len(avg_scores) - window + 1)]
        gen_numbers = generations[window-1:]
        plt.plot(gen_numbers, moving_avg, linewidth=2,
                label=f'Moving Average (window={window})', color='blue')

    # Plot all average scores
    plt.plot(generations, avg_scores, linewidth=1, alpha=0.5,
            label='Average Score', color='lightblue')

    plt.title('Genetic Algorithm Training - Average Score per Generation',
             fontsize=14, fontweight='bold')
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Average Score', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Save the plot
    output_file = 'genetic_averagescore.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved average score plot to {output_file}")
    plt.close()

    # Plot 2: Best Score
    plt.figure(figsize=(10, 6))
    plt.plot(generations, best_scores, marker='o', linewidth=2,
            markersize=4, color='red', label='Best Score', alpha=0.7)
    plt.title('Genetic Algorithm Training - Best Score per Generation',
             fontsize=14, fontweight='bold')
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Best Score', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Save the plot
    output_file = 'genetic_maxscore.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved max score plot to {output_file}")
    plt.close()

    # Plot 3: Time per Generation
    plt.figure(figsize=(10, 6))
    plt.plot(generations, times, linewidth=2, color='green', label='Time per Generation', alpha=0.7)

    # Calculate moving average for time
    if len(times) >= window:
        moving_avg_time = [sum(times[i:i+window]) / window
                          for i in range(len(times) - window + 1)]
        gen_numbers = generations[window-1:]
        plt.plot(gen_numbers, moving_avg_time, linewidth=2,
                label=f'Moving Average Time (window={window})', color='darkgreen', linestyle='--')

    plt.title('Genetic Algorithm Training - Time per Generation',
             fontsize=14, fontweight='bold')
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Save the plot
    output_file = 'genetic_time.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved time plot to {output_file}")
    plt.close()

    # Plot 4: Cumulative Time
    cumulative_times = []
    cumulative = 0
    for time in times:
        cumulative += time
        cumulative_times.append(cumulative)

    plt.figure(figsize=(10, 6))
    plt.plot(generations, cumulative_times, linewidth=2, color='purple', label='Cumulative Time')
    plt.title('Genetic Algorithm Training - Cumulative Time',
             fontsize=14, fontweight='bold')
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Cumulative Time (seconds)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Save the plot
    output_file = 'genetic_cumulative_time.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved cumulative time plot to {output_file}")
    plt.close()

    # Print summary statistics
    total_time = sum(times)
    avg_time = sum(times) / len(times) if times else 0
    max_time = max(times) if times else 0
    min_time = min(times) if times else 0

    print(f"\nTraining Summary:")
    print(f"  Total Generations: {len(generations)}")
    print(f"  Final Best Score: {best_scores[-1]:.2f}")
    print(f"  Final Average Score: {avg_scores[-1]:.2f}")
    print(f"  Overall Best Score: {max(best_scores):.2f}")
    print(f"  Overall Best Average: {max(avg_scores):.2f}")
    print(f"\nTime Statistics:")
    print(f"  Total Time: {total_time:.1f}s ({total_time/60:.1f} minutes, {total_time/3600:.2f} hours)")
    print(f"  Average Time per Generation: {avg_time:.1f}s")
    print(f"  Max Time per Generation: {max_time:.1f}s")
    print(f"  Min Time per Generation: {min_time:.1f}s")

if __name__ == "__main__":
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(script_dir, 'genetic_gen.txt')

    if os.path.exists(log_file):
        plot_genetic_training(log_file)
    else:
        print(f"Error: {log_file} not found!")
        print("Please make sure genetic_gen.txt is in the same directory as this script.")

