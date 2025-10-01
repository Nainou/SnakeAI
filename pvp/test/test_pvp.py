"""
Testing Script for PvP Snake AI

Loads trained models and runs PvP matches with Pygame rendering
and visual neural network view for each snake.
"""

import sys
import os
import time
import torch
import numpy as np
import pygame
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from game.snake_game_pvp import SnakeGamePvP
from ai.genetic_algorithm import GeneticAlgorithm


def visualize_neural_network(network, snake_id, screen, x, y, width, height):
    """
    Visualize a neural network on the pygame screen

    Args:
        network: The neural network to visualize
        snake_id: ID of the snake (for color coding)
        screen: Pygame screen surface
        x, y: Position to draw the network
        width, height: Size of the visualization area
    """
    # Snake colors for network visualization
    snake_colors = [
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 0, 0),    # Red
        (255, 255, 0)   # Yellow
    ]

    color = snake_colors[snake_id % len(snake_colors)]

    # Get network structure
    layers = []
    for name, module in network.named_modules():
        if isinstance(module, torch.nn.Linear):
            layers.append(module.weight.shape[0])

    if not layers:
        return

    # Calculate positions for nodes
    layer_width = width // len(layers)
    node_radius = min(layer_width // 4, 15)

    # Draw connections
    for i in range(len(layers) - 1):
        current_layer_size = layers[i]
        next_layer_size = layers[i + 1]

        current_x = x + i * layer_width + layer_width // 2
        next_x = x + (i + 1) * layer_width + layer_width // 2

        current_y_start = y + height // 2 - (current_layer_size - 1) * 20 // 2
        next_y_start = y + height // 2 - (next_layer_size - 1) * 20 // 2

        for j in range(current_layer_size):
            current_y = current_y_start + j * 20
            for k in range(next_layer_size):
                next_y = next_y_start + k * 20

                # Draw connection line
                pygame.draw.line(screen, (100, 100, 100),
                               (current_x, current_y), (next_x, next_y), 1)

    # Draw nodes
    for i, layer_size in enumerate(layers):
        layer_x = x + i * layer_width + layer_width // 2
        layer_y_start = y + height // 2 - (layer_size - 1) * 20 // 2

        for j in range(layer_size):
            node_y = layer_y_start + j * 20

            # Draw node
            pygame.draw.circle(screen, color, (layer_x, node_y), node_radius)
            pygame.draw.circle(screen, (0, 0, 0), (layer_x, node_y), node_radius, 2)

    # Draw layer labels
    font = pygame.font.Font(None, 24)
    for i, layer_size in enumerate(layers):
        layer_x = x + i * layer_width + layer_width // 2
        label = font.render(f"L{i+1}", True, (0, 0, 0))
        label_rect = label.get_rect(center=(layer_x, y - 20))
        screen.blit(label, label_rect)


def test_pvp_models(model_paths, num_games=5, grid_size=20, num_snakes=2, render_delay=100):
    """
    Test PvP models with visualization

    Args:
        model_paths: List of model file paths (one per snake)
        num_games: Number of games to play
        grid_size: Size of the game grid
        num_snakes: Number of snakes (2-4)
        render_delay: Delay between frames in milliseconds
    """
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

    # Initialize pygame for visualization
    pygame.init()

    # Create larger window to accommodate network visualizations
    window_width = 1000
    window_height = 800
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("PvP Snake AI - Neural Network Visualization")
    clock = pygame.time.Clock()

    # Game area dimensions
    game_width = min(600, window_width - 400)
    game_height = min(600, window_height - 200)
    game_x = 20
    game_y = 20

    # Network visualization areas
    network_width = 150
    network_height = 200
    network_y = game_y + game_height + 20

    scores = {i: [] for i in range(num_snakes)}
    wins = {i: 0 for i in range(num_snakes)}

    print(f"\nStarting PvP testing with {num_snakes} snakes")
    print("=" * 50)

    for game_num in range(num_games):
        print(f"\nGame {game_num + 1}/{num_games}")

        # Create game
        game = SnakeGamePvP(grid_size=grid_size, num_snakes=num_snakes, display=False)
        game.set_snake_networks(networks)
        states = game.reset()

        # Debug: Check networks after reset (uncomment for debugging)
        # print(f"Debug - Networks after reset: {[snake.network is not None for snake in game.snakes if snake.alive]}")
        # print(f"Debug - States keys: {list(states.keys())}")
        # print(f"Debug - Networks keys: {list(networks.keys())}")

        # Game loop
        running = True
        step = 0
        while running and not game.done:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        break

            if not running:
                break

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

            # Render the game
            screen.fill((255, 255, 255))

            # Draw game area
            pygame.draw.rect(screen, (200, 200, 200), (game_x, game_y, game_width, game_height))

            # Draw grid
            square_size = min(game_width // grid_size, game_height // grid_size)
            for i in range(grid_size + 1):
                pygame.draw.line(screen, (150, 150, 150),
                               (game_x + i * square_size, game_y),
                               (game_x + i * square_size, game_y + game_height), 1)
                pygame.draw.line(screen, (150, 150, 150),
                               (game_x, game_y + i * square_size),
                               (game_x + game_width, game_y + i * square_size), 1)

            # Draw food
            if game.food_position:
                food_x = game_x + game.food_position[0] * square_size
                food_y = game_y + game.food_position[1] * square_size
                pygame.draw.rect(screen, (255, 0, 0),
                               (food_x, food_y, square_size, square_size))
                pygame.draw.rect(screen, (0, 0, 0),
                               (food_x, food_y, square_size, square_size), 2)

            # Draw snakes
            for snake in game.snakes:
                if not snake.alive:
                    continue

                for i, position in enumerate(snake.positions):
                    snake_x = game_x + position[0] * square_size
                    snake_y = game_y + position[1] * square_size

                    # Head is brighter, body is darker
                    if i == 0:  # Head
                        color = snake.color
                    else:  # Body
                        color = tuple(max(0, c - 50) for c in snake.color)

                    pygame.draw.rect(screen, color,
                                   (snake_x, snake_y, square_size, square_size))
                    pygame.draw.rect(screen, (0, 0, 0),
                                   (snake_x, snake_y, square_size, square_size), 1)

            # Draw neural network visualizations
            for i in range(num_snakes):
                if i in networks:
                    network_x = game_x + game_width + 20 + (i % 2) * (network_width + 20)
                    network_y_pos = network_y + (i // 2) * (network_height + 20)

                    # Draw network background
                    pygame.draw.rect(screen, (240, 240, 240),
                                   (network_x, network_y_pos, network_width, network_height))
                    pygame.draw.rect(screen, (0, 0, 0),
                                   (network_x, network_y_pos, network_width, network_height), 2)

                    # Draw network
                    visualize_neural_network(networks[i], i, screen,
                                           network_x + 10, network_y_pos + 10,
                                           network_width - 20, network_height - 20)

                    # Draw snake label
                    font = pygame.font.Font(None, 24)
                    label = font.render(f"Snake {i}", True, game.snake_colors[i])
                    screen.blit(label, (network_x + 5, network_y_pos - 25))

            # Draw scores
            font = pygame.font.Font(None, 24)
            y_offset = game_y + game_height + 20
            for snake in game.snakes:
                score_text = font.render(f"Snake {snake.snake_id}: {snake.score}", True, snake.color)
                screen.blit(score_text, (game_x, y_offset))
                y_offset += 25

            # Draw steps
            steps_text = font.render(f"Steps: {game.steps}", True, (0, 0, 0))
            screen.blit(steps_text, (game_x, y_offset))

            # Draw winner if game is done
            if game.done and game.winner is not None:
                winner_text = font.render(f"Winner: Snake {game.winner}!", True, (0, 0, 0))
                screen.blit(winner_text, (game_x, y_offset + 30))

            pygame.display.flip()
            clock.tick(1000 // render_delay)

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

    pygame.quit()

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
    """Main testing function"""
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
            print("❌ No valid models selected")
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
        print(f"❌ Invalid selection: {e}")
    except KeyboardInterrupt:
        print("\n\nTesting interrupted by user.")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
