#!/usr/bin/env python3
"""
PvP Snake AI Demo

Quick demonstration of the PvP snake game with random AI.
"""

import sys
import time
import random
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from game.snake_game_pvp import SnakeGamePvP
from ai.neural_network import NeuralNetwork


def random_ai_demo():
    """Demo with random AI snakes"""
    print("PvP Snake AI Demo - Random AI")
    print("=" * 40)

    # Create game with 2 snakes
    game = SnakeGamePvP(grid_size=15, num_snakes=2, display=True, render_delay=10)

    # Create random neural networks for each snake
    networks = {}
    for i in range(2):
        networks[i] = NeuralNetwork()

    game.set_snake_networks(networks)
    states = game.reset()

    print("Starting game with neural network AI...")
    print("Press ESC or close window to exit")

    # Game loop
    while not game.done:
        # Get actions from neural networks
        actions = {}
        for snake_id, state in states.items():
            if snake_id in networks:
                actions[snake_id] = networks[snake_id].act(state)
            else:
                actions[snake_id] = random.randint(0, 2)  # Random action as fallback

        # Step the game
        states, rewards, done, info = game.step(actions)

        # Render
        game.render()

        # Check for quit
        if game.done:
            break

    # Show results
    scores = game.get_snake_scores()
    print(f"\nGame Over!")
    print(f"Final Scores: {scores}")
    if game.winner is not None:
        print(f"Winner: Snake {game.winner}")
    else:
        print("No winner (timeout)")

    game.close()


def main():
    """Main demo function"""
    try:
        random_ai_demo()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure you have pygame installed: pip install pygame")


if __name__ == "__main__":
    main()
