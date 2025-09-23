# 🐍 Genetic Algorithm Snake AI

This implementation uses a **genetic algorithm** to evolve neural networks that play Snake, replacing the original Deep Q-Learning (DQN) approach with evolutionary computation.

## 🧬 How Genetic Algorithms Work

Instead of learning through trial and error like DQN, genetic algorithms evolve a population of neural networks over multiple generations:

1. **Population**: Start with random neural networks (individuals)
2. **Evaluation**: Test each network by playing Snake games
3. **Selection**: Choose the best performers as parents
4. **Crossover**: Combine parent networks to create offspring
5. **Mutation**: Add random changes to introduce diversity
6. **Evolution**: Repeat for multiple generations

## 🎯 Key Features

- **Population-based**: Evolves 20-50 neural networks simultaneously
- **Generation-based**: Improves over 20-100 generations
- **Fitness evaluation**: Networks are ranked by game performance
- **Elitism**: Best networks are preserved across generations
- **Tournament selection**: Competitive selection of parents
- **Crossover**: Single-point crossover of network weights
- **Mutation**: Gaussian noise added to weights

## 📊 Fitness Function

Networks are evaluated based on:
- **Average score** (primary factor)
- **Maximum score** achieved
- **Win rate** (completing the game)
- **Efficiency** (steps taken)

```python
fitness = (avg_score * 100 +
          max_score * 50 +
          win_rate * 1000 +
          efficiency_bonus)
```

## 🚀 Quick Start

### 1. Train a New Population
```bash
python train_genetic.py
```

### 2. Test Trained Networks
```bash
python test.py
```

### 3. Compare DQN vs Genetic Algorithm
```bash
python test.py
# Choose option 3 for comparison
```

## 📁 Files

- `genetic_snake.py` - Main genetic algorithm implementation
- `train_genetic.py` - Training script with user interface
- `test.py` - Testing and comparison script
- `snake_game.py` - Snake game environment (unchanged)

## ⚙️ Configuration

### Population Parameters
- **Population size**: 20-50 individuals
- **Generations**: 20-100 generations
- **Games per evaluation**: 3-5 games

### Genetic Operators
- **Mutation rate**: 0.1 (10% of weights mutated)
- **Crossover rate**: 0.7 (70% crossover probability)
- **Elitism ratio**: 0.1 (top 10% preserved)
- **Tournament size**: 3 individuals

### Neural Network
- **Input**: 17 features (same as DQN)
- **Hidden layers**: 32 → 16 neurons
- **Output**: 3 actions (straight, left, right)
- **Architecture**: Smaller than DQN for faster evolution

## 📈 Training Progress

The algorithm tracks and visualizes:
- **Fitness evolution** over generations
- **Score improvement** over time
- **Population diversity** in final generation
- **Performance statistics**

## 🎮 Game Performance

Genetic algorithms typically achieve:
- **Average scores**: 3-8 points
- **Maximum scores**: 10-20+ points
- **Convergence**: 15-30 generations
- **Consistency**: More stable than DQN

## 🔄 Advantages vs DQN

### Genetic Algorithm
- ✅ No hyperparameter tuning (learning rate, epsilon, etc.)
- ✅ Population diversity prevents local optima
- ✅ More interpretable evolution process
- ✅ No replay buffer or target networks needed
- ✅ Naturally handles exploration vs exploitation

### DQN
- ✅ Faster training per episode
- ✅ Can learn online during gameplay
- ✅ Better sample efficiency
- ✅ More established in game AI

## 🔧 Customization

### Modify Genetic Operators
```python
# In genetic_snake.py
ga = GeneticAlgorithm(
    population_size=50,
    mutation_rate=0.15,    # Higher mutation
    crossover_rate=0.8,    # More crossover
    elitism_ratio=0.2      # Keep more elites
)
```

### Change Network Architecture
```python
# In NeuralNetwork class
def __init__(self, input_size=17, hidden_size=64, output_size=3):
    # Larger networks for more complex behaviors
```

### Adjust Fitness Function
```python
# In evaluate_individual method
fitness = (avg_score * 100 +       # Score weight
          max_score * 50 +         # Peak performance
          win_rate * 1000 +        # Completion bonus
          (avg_steps * -0.1))      # Efficiency penalty
```

## 🎯 Expected Results

After 20-30 generations with a population of 30:
- Networks learn to avoid walls and self-collision
- Develop food-seeking behavior
- Achieve consistent scores of 5-10 points
- Best individuals may reach 15-25 points

## 🐛 Troubleshooting

### Low Performance
- Increase population size
- Add more generations
- Reduce mutation rate
- Increase games per evaluation

### Slow Training
- Reduce population size
- Fewer games per evaluation
- Smaller neural networks
- Disable visualization during training

### No Improvement
- Check fitness function
- Increase mutation rate
- Verify game environment
- Try different selection methods

## 📚 Further Reading

- [Genetic Algorithms in Neural Networks](https://en.wikipedia.org/wiki/Neuroevolution)
- [NEAT Algorithm](http://www.cs.ucf.edu/~kstanley/neat.html)
- [Evolutionary Strategies](https://blog.openai.com/evolution-strategies/)

---

**Happy evolving!** 🧬🐍
