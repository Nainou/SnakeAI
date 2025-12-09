# SnakeAI

A collection of AI approaches for training neural networks to play Snake using different state representations and learning algorithms.

## Project Structure

### [ray/](ray/)
**Raycast-based State Representation with DQN**

21-dimensional feature vector (food direction, current direction, danger detection, distances). DQN (128→128→64) trained with Deep Q-Learning.

**Files:** `snake_game_ray.py`, `train_model_ray.py`, `test_model_ray.py`

### [pixel/](pixel/)
**Pixel-based State Representation with Convolutional Q-Network**

5-channel grid representation (head, body, tail, food, walls). Convolutional Q-Network (16→32→64 feature maps).

**Files:** `snake_game_pixel.py`, `train_model_pixel.py`, `test_model_pixel.py`

### [genetic/](genetic/)
**Genetic Algorithm Evolution**

32-feature vector similar to raycast. (20→12) Network evolved using genetic algorithms (500 parents, 1000 offspring) with tournament selection.

**Files:** `snake_game_genetic.py`, `train_model_genetic.py`, `test_model_genetic.py`

### [pvp/](pvp/)
**Player vs Player Snake Competition**

2-4 genetically trained snakes compete on a 20×20 grid. Each snake uses a 17-feature neural network (64 neurons) trained with genetic algorithms.

**Files:** `game/snake_game_pvp.py`, `train.py`, `test.py`

### [demo/](demo/)
**Demo App**

GUI for testing and comparing models. Supports single-player and PvP modes with replay functionality.

**Usage:** Place trained models to folder -> `demo/models`, start demo.py


## Requirements

Python 3.7+, PyTorch, NumPy, Pygame, Matplotlib, Tkinter

```bash
pip install -r requirements.txt
```

## Quick Start

**Training:**
- Ray: `cd ray && python train_model_ray.py`
- Pixel: `cd pixel && python train_model_pixel.py`
- Genetic: `cd genetic && python train_model_genetic.py`
- PvP: `cd pvp && python train.py`

## Model Naming

The Demo requires model name to have the model type, input size, hidden layer size and output size.

- Ray: `ray_21_128_3_ep####.pth`
- Pixel: `pixel_72_128_3_ep####.pth`
- Genetic: `genetic_32_20_12_4_gen###.pth`
- PvP: `pvp_snake_gen_###.pth`
