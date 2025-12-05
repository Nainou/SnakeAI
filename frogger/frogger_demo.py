from train_frogger_ray import DQNAgent, test_agent
from frogger_game_ray import FroggerGameRL

if __name__ == "__main__":
    # Initialize game to get state and action sizes
    game = FroggerGameRL(grid_width=10, grid_height=12, display=False)
    state_size = game.get_state_size()  # Should be 18
    action_size = 5  # stay, up, down, left, right

    # Create agent with matching architecture
    agent = DQNAgent(state_size=state_size, action_size=action_size, lr=0.001)

    # Load the trained model
    model_path = "frogger/saved/frogger_18_128_5.pth"
    agent.load(model_path)
    print(f"Loaded model from {model_path}")

    # Test the agent
    test_agent(agent, num_games=10, display=True)