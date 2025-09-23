import pygame
import sys
import numpy as np
import os

def test_dqn_agent():
    """Test the original DQN agent"""
    from snakeAI import DQNAgent

    def test_agent(agent, num_games=10, display=True):
        from snake_game import SnakeGameRL

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

        print(f"\nDQN Results:")
        print(f"Average Score: {np.mean(scores):.2f}")
        print(f"Max Score: {max(scores)}")
        print(f"Min Score: {min(scores)}")

        return scores

    agent = DQNAgent(state_size=17, action_size=3)
    try:
        agent.load("snake_model_final.pth")
        return test_agent(agent, num_games=5, display=True)
    except Exception as e:
        print(f"Error loading DQN model: {e}")
        return []

def test_genetic_agent():
    """Test the genetic algorithm agent"""
    from genetic_snake import test_genetic_individual

    # Try to find the best genetic model
    model_files = [
        'genetic_snake_final.pth',
        'genetic_snake_gen_20.pth',
        'genetic_snake_gen_10.pth'
    ]

    for model_file in model_files:
        if os.path.exists(model_file):
            print(f"Testing genetic algorithm model: {model_file}")
            return test_genetic_individual(model_file, num_games=5, display=True)

    print("No genetic algorithm models found. Please train one first.")
    return []

def compare_models():
    """Compare DQN and Genetic Algorithm performance"""
    print("=" * 60)
    print("SNAKE AI MODEL COMPARISON")
    print("=" * 60)

    print("\n1. Testing DQN Agent...")
    print("-" * 30)
    dqn_scores = test_dqn_agent()

    print("\n2. Testing Genetic Algorithm Agent...")
    print("-" * 30)
    genetic_scores = test_genetic_agent()

    if dqn_scores and genetic_scores:
        print("\n" + "=" * 60)
        print("COMPARISON RESULTS")
        print("=" * 60)
        print(f"DQN Average Score: {np.mean(dqn_scores):.2f} ± {np.std(dqn_scores):.2f}")
        print(f"Genetic Average Score: {np.mean(genetic_scores):.2f} ± {np.std(genetic_scores):.2f}")
        print(f"DQN Max Score: {max(dqn_scores)}")
        print(f"Genetic Max Score: {max(genetic_scores)}")

        if np.mean(genetic_scores) > np.mean(dqn_scores):
            print("\n🏆 Genetic Algorithm performs better on average!")
        elif np.mean(dqn_scores) > np.mean(genetic_scores):
            print("\n🏆 DQN performs better on average!")
        else:
            print("\n🤝 Both models perform similarly!")

if __name__ == "__main__":
    print("Snake AI Testing Suite")
    print("Choose an option:")
    print("1. Test DQN Agent only")
    print("2. Test Genetic Algorithm only")
    print("3. Compare both models")
    print("4. Train new Genetic Algorithm")

    choice = input("\nEnter choice (1-4): ").strip()

    if choice == "1":
        test_dqn_agent()
    elif choice == "2":
        test_genetic_agent()
    elif choice == "3":
        compare_models()
    elif choice == "4":
        print("\nQuick training with GPU + threading...")
        from genetic_snake import train_genetic_algorithm, plot_evolution_progress
        import multiprocessing
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        threads = min(max(2, multiprocessing.cpu_count() - 1), 6)
        print(f"Using device: {device}")
        ga = train_genetic_algorithm(
            generations=15,
            population_size=20,
            games_per_eval=3,
            num_threads=threads,
            quiet=True,
            verbose=False,
            device=device
        )
        print(f"Training complete! Best score: {ga.best_score_history[-1]:.1f}")
        print("You can now test the genetic algorithm.")
    else:
        print("Invalid choice. Testing genetic algorithm by default...")
        test_genetic_agent()