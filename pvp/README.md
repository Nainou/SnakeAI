# PvP Snake AI Project

A Python project where 2–4 genetically trained snakes compete on a large grid using neural networks trained with genetic algorithms.

## Features

- **PvP Competition**: 2-4 snakes compete simultaneously on a large grid
- **Genetic Algorithm Training**: Neural networks evolve through genetic algorithms
- **Distinct Colors**: Each snake has a unique color for easy identification
- **Modular Architecture**: Clean separation of game logic, AI, training, and testing
- **Visual Neural Networks**: Real-time visualization of each snake's neural network
- **Headless Training**: Fast training without rendering for maximum speed

## Project Structure

```
pvp/
├── game/                    # Game logic and environment
│   ├── __init__.py
│   ├── snake.py            # Individual snake class
│   └── snake_game_pvp.py   # PvP game implementation
├── ai/                      # AI components
│   ├── __init__.py
│   ├── neural_network.py   # Neural network architecture
│   └── genetic_algorithm.py # Genetic algorithm implementation
├── train/                   # Training scripts
│   ├── __init__.py
│   └── train_pvp.py        # Headless training
├── test/                    # Testing and visualization
│   ├── __init__.py
│   └── test_pvp.py         # Testing with visualization
├── saved/                   # Saved model files (.pth)
├── train.py                 # Training entry point
├── test.py                  # Testing entry point
└── README.md               # This file
```

## Quick Start

### 1. Training

Train the genetic algorithm (headless, fast):

```bash
python train.py
```

Choose from:
- Quick train (20 generations, 30 population, 2 snakes)
- Medium train (40 generations, 50 population, 2 snakes)
- Full train (60 generations, 80 population, 4 snakes)
- Custom configuration

### 2. Testing

Test trained models with visualization:

```bash
python test.py
```

Features:
- Pygame rendering of the game
- Real-time neural network visualization for each snake
- Multiple model selection
- Performance statistics

## Game Rules

- **Classic Snake**: Eat food to grow, avoid walls and collisions
- **PvP Competition**: Last snake alive wins
- **Collision Detection**: Hitting walls, self, or other snakes = death
- **Food System**: Eating food increases score and length
- **Large Grid**: 20x20 grid for complex strategies

## AI Architecture

### Neural Network
- **Input**: 17 features (food direction, current direction, danger detection, distances)
- **Architecture**: 4-layer fully connected network with ReLU activations
- **Output**: 3 actions (straight, turn right, turn left)
- **Regularization**: Dropout for better generalization

### Genetic Algorithm
- **Population**: 30-80 individuals per generation
- **Selection**: Tournament selection for parent selection
- **Crossover**: Single-point crossover with 80% probability
- **Mutation**: Adaptive mutation rate (15% base rate)
- **Elitism**: Keep best 15% of population unchanged
- **Fitness**: Multi-factor scoring including wins, scores, and positions

## Training Process

1. **Initialization**: Create random population of neural networks
2. **Evaluation**: Each individual plays multiple PvP games
3. **Fitness Calculation**: Score based on wins, average score, and position
4. **Selection**: Choose parents using tournament selection
5. **Reproduction**: Create offspring through crossover and mutation
6. **Evolution**: Replace population with new generation
7. **Repeat**: Continue for specified number of generations

## Testing Features

### Visual Neural Network
- Real-time display of each snake's neural network
- Node connections and layer structure
- Color-coded by snake ID
- Updates during gameplay

### Performance Metrics
- Average scores per snake
- Win rates
- Maximum scores achieved
- Position statistics

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- Pygame
- Matplotlib

## Installation

```bash
pip install torch numpy pygame matplotlib
```

## Usage Examples

### Custom Training
```python
from train.train_pvp import train_pvp_genetic_algorithm

ga = train_pvp_genetic_algorithm(
    generations=50,
    population_size=60,
    games_per_eval=5,
    num_snakes=3,
    verbose=True
)
```

### Custom Testing
```python
from test.test_pvp import test_pvp_models

scores, wins = test_pvp_models(
    model_paths=['saved/model1.pth', 'saved/model2.pth'],
    num_games=10,
    num_snakes=2
)
```

## File Descriptions

- **`game/snake.py`**: Individual snake class with neural network integration
- **`game/snake_game_pvp.py`**: Main PvP game logic and rendering
- **`ai/neural_network.py`**: Neural network architecture
- **`ai/genetic_algorithm.py`**: Genetic algorithm implementation
- **`train/train_pvp.py`**: Headless training script
- **`test/test_pvp.py`**: Testing with visualization