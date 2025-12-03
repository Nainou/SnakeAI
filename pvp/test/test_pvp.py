# Testing Script for PvP Snake AI
# Loads trained models and runs PvP matches with Pygame rendering
# and visual neural network view for each snake.

import sys
import os
import time
import torch
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from game.snake_game_pvp import SnakeGamePvP
from ai.genetic_algorithm import GeneticAlgorithm


# Note: The visualize_neural_network function has been removed as the game
# now has built-in neural network visualization. Click on snake names to view networks.


def test_pvp_models(model_paths, num_games=5, grid_size=20, num_snakes=2, render_delay=10):
    # Test PvP models with visualization
    # Args:
    #   model_paths: List of model file paths (one per snake)
    #   num_games: Number of games to play
    #   grid_size: Size of the game grid
    #   num_snakes: Number of snakes (2-4)
    #   render_delay: Delay between frames in milliseconds
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    ga = GeneticAlgorithm(device=device)
    networks = {}

    for i, model_path in enumerate(model_paths[:num_snakes]):
        if os.path.exists(model_path):
            individual = ga.load_individual(model_path)
            networks[i] = individual.network
            print(f"Loaded model for Snake {i}: {model_path}")
        else:
            print(f"Warning: Model not found: {model_path}")
            # Create a random network as fallback
            from ai.neural_network import NeuralNetwork
            networks[i] = NeuralNetwork(device=device)
            print(f"Created random network for Snake {i}")

    # Ensure we have networks for all snakes
    for i in range(num_snakes):
        if i not in networks:
            from ai.neural_network import NeuralNetwork
            networks[i] = NeuralNetwork(device=device)
            print(f"Created random network for Snake {i}")

    # Debug output (uncomment for debugging)
    # print(f"\nFinal networks: {list(networks.keys())}")
    # print(f"Number of snakes: {num_snakes}")

    # Note: The game now handles its own rendering with neural network visualization
    # Just set display=True and use game.render()

    scores = {i: [] for i in range(num_snakes)}
    wins = {i: 0 for i in range(num_snakes)}

    print(f"\nStarting PvP testing with {num_snakes} snakes")
    print("=" * 50)

    for game_num in range(num_games):
        print(f"\nGame {game_num + 1}/{num_games}")

        # Create game with display enabled to use built-in rendering
        game = SnakeGamePvP(grid_size=grid_size, num_snakes=num_snakes, display=True, render_delay=render_delay)
        game.set_snake_networks(networks)
        states = game.reset()

        # Debug: Check networks after reset (uncomment for debugging)
        # print(f"Debug - Networks after reset: {[snake.network is not None for snake in game.snakes if snake.alive]}")
        # print(f"Debug - States keys: {list(states.keys())}")
        # print(f"Debug - Networks keys: {list(networks.keys())}")

        print("Click on a snake name to view its neural network!")
        print("Press ESC or close window to exit")

        # Game loop
        running = True
        step = 0
        while running and not game.done:
            # Get actions from neural networks
            actions = {}
            for snake_id, state in states.items():
                if snake_id in networks:
                    action = networks[snake_id].act(state)
                    actions[snake_id] = action
                    # Debug: print action for each snake (uncomment for debugging)
                    # if step < 5:  # Only print first 5 steps
                    #     print(f"Snake {snake_id}: action={action}")
                else:
                    print(f"Warning: No network for snake {snake_id}")
                    # Create a random action as fallback
                    actions[snake_id] = 0  # Go straight

            # Step the game
            states, rewards, done, info = game.step(actions)
            step += 1

            # Render the game (now includes neural network visualization)
            game.render()

            # Check if window was closed
            if game.done:
                break

        # Record results
        snake_scores = game.get_snake_scores()
        for snake_id, score in snake_scores.items():
            scores[snake_id].append(score)

        if game.winner is not None:
            wins[game.winner] += 1

        print(f"  Scores: {snake_scores}")
        if game.winner is not None:
            print(f"  Winner: Snake {game.winner}")
        else:
            print("  No winner (timeout)")

        # Close the game window
        game.close()

    # Print final results
    print(f"\nFinal Results:")
    print("=" * 30)
    for i in range(num_snakes):
        if scores[i]:
            avg_score = np.mean(scores[i])
            max_score = max(scores[i])
            win_rate = wins[i] / num_games * 100
            print(f"Snake {i}: Avg={avg_score:.1f}, Max={max_score}, Wins={wins[i]}/{num_games} ({win_rate:.1f}%)")

    return scores, wins


def main():
    # Main testing function
    print("PvP Snake AI Testing")
    print("=" * 30)

    # Check for saved models
    saved_dir = Path("pvp/saved")
    if not saved_dir.exists():
        print("No saved models found. Train models first!")
        return

    # Find available models
    model_files = list(saved_dir.glob("*.pth"))
    if not model_files:
        print("No .pth model files found in saved/ directory")
        return

    print(f"Found {len(model_files)} model files:")
    for i, model_file in enumerate(model_files):
        print(f"  {i}: {model_file.name}")

    # Get user input for model selection
    try:
        if len(model_files) == 1:
            selected_models = [str(model_files[0])]
            print(f"Using only available model: {model_files[0].name}")
        else:
            print("\nSelect models to use (comma-separated indices, e.g., 0,1,2):")
            selection = input("Selection: ").strip()
            indices = [int(x.strip()) for x in selection.split(',')]
            selected_models = [str(model_files[i]) for i in indices if 0 <= i < len(model_files)]

        if not selected_models:
            print("No valid models selected")
            return

        # Get game configuration
        num_snakes = min(len(selected_models), 4)
        if len(selected_models) > 4:
            print(f"Warning: Only using first 4 models")
            selected_models = selected_models[:4]

        print(f"\nUsing {num_snakes} snakes with models:")
        for i, model in enumerate(selected_models):
            print(f"  Snake {i}: {Path(model).name}")

        # Run tests
        scores, wins = test_pvp_models(selected_models, num_games=5, num_snakes=num_snakes)

    except (ValueError, IndexError) as e:
        print(f"Invalid selection: {e}")
    except KeyboardInterrupt:
        print("\n\nTesting interrupted by user.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
