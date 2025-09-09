import pygame
import sys
from pixel.snakeAI_pixel import DQNAgent
import numpy as np

def test_agent(agent, num_games=10, display=True):
    from snake_game_pixel import SnakeGameRL

    game = SnakeGameRL(grid_size=10, display=display, render_delay=30)
    agent.epsilon = 0  # No exploration during testing

    scores = []

    for i in range(num_games):
        state = game.reset()

        while not game.done:
            action = agent.act(state)
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

    print(f"\nAverage Score: {np.mean(scores):.2f}")
    print(f"Max Score: {max(scores)}")
    print(f"Min Score: {min(scores)}")

    return scores


if __name__ == "__main__":
    agent = DQNAgent(state_size=200, action_size=3)
    agent.load("pixel/saved/snake_model_episode_pixel_400.pth")
    test_agent(agent, num_games=5, display=True)