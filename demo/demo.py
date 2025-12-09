import sys
import os
import torch
import numpy as np
import re
import threading
import pickle
import json
from collections import deque
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datetime import datetime

from snake_game_single import SnakeGameSingle, Direction
from snake_game_pvp import SnakeGamePvP
from neural_network import NeuralNetwork, ConvQN

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext


class ModelMetadata:

    def __init__(self, model_type: str, input_size: int, hidden_size: int,
                 output_size: int, extra_info: str = "", hidden_size2: Optional[int] = None):
        self.model_type = model_type  # ray,pixel,genetic,pvp
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_size2 = hidden_size2  # genetic models only
        self.output_size = output_size
        self.extra_info = extra_info  # genetic models only

    def to_filename(self, prefix: str = "") -> str:
        if self.hidden_size2 is not None:
            # Genetic model format: genetic_32_20_12_4_extra
            parts = [self.model_type, str(self.input_size), str(self.hidden_size),
                     str(self.hidden_size2), str(self.output_size)]
        else:
            # Standard format: type_input_hidden_output_extra
            parts = [self.model_type, str(self.input_size), str(self.hidden_size),
                     str(self.output_size)]
        if self.extra_info:
            parts.append(self.extra_info)
        filename = "_".join(parts) + ".pth"
        if prefix:
            filename = prefix + "_" + filename
        return filename

    def __repr__(self):
        if self.hidden_size2 is not None:
            return (f"ModelMetadata(type={self.model_type}, "
                    f"input={self.input_size}, hidden1={self.hidden_size}, "
                    f"hidden2={self.hidden_size2}, output={self.output_size}, extra='{self.extra_info}')")
        else:
            return (f"ModelMetadata(type={self.model_type}, "
                    f"input={self.input_size}, hidden={self.hidden_size}, "
                    f"output={self.output_size}, extra='{self.extra_info}')")


def parse_model_filename(filename: str) -> Optional[ModelMetadata]:
    # Remove .pth extension
    name = Path(filename).stem

    # Pattern for genetic models: genetic_input_hidden1_hidden2_output_optional_extra
    genetic_pattern = r'^genetic_(\d+)_(\d+)_(\d+)_(\d+)(?:_(.+))?$'
    match = re.match(genetic_pattern, name)
    if match:
        model_type = "genetic"
        input_size = int(match.group(1))
        hidden_size = int(match.group(2))
        hidden_size2 = int(match.group(3))
        output_size = int(match.group(4))
        extra_info = match.group(5) or ""
        return ModelMetadata(model_type, input_size, hidden_size, output_size, extra_info, hidden_size2)

    pattern = r'^([a-z]+)_(\d+)_(\d+)_(\d+)(?:_(.+))?$'
    match = re.match(pattern, name)

    if match:
        model_type = match.group(1)
        input_size = int(match.group(2))
        hidden_size = int(match.group(3))
        output_size = int(match.group(4))
        extra_info = match.group(5) or ""

        return ModelMetadata(model_type, input_size, hidden_size, output_size, extra_info)

    legacy_pattern = r'^([a-z_]+)_gen_(\d+)$'
    match = re.match(legacy_pattern, name)
    if match:
        model_type = "pvp"
        input_size = 17
        hidden_size = 64
        output_size = 3
        extra_info = f"gen{match.group(2)}"
        return ModelMetadata(model_type, input_size, hidden_size, output_size, extra_info)
    return None


def create_model_filename(model_type: str, input_size: int, hidden_size: int,
                         output_size: int, extra_info: str = "",
                         prefix: str = "") -> str:
    metadata = ModelMetadata(model_type, input_size, hidden_size, output_size, extra_info)
    return metadata.to_filename(prefix)


def list_models_in_directory(directory: Path) -> Dict[str, ModelMetadata]:
    models = {}

    if not directory.exists():
        return models

    for file_path in directory.glob("*.pth"):
        metadata = parse_model_filename(file_path.name)
        if metadata:
            models[file_path.name] = metadata
        else:
            models[file_path.name] = None

    return models


def format_model_info(metadata: Optional[ModelMetadata], filename: str) -> str:
    if metadata:
        if metadata.hidden_size2 is not None:
            info = (f"Type: {metadata.model_type.upper()}, "
                    f"Input: {metadata.input_size}, "
                    f"Hidden1: {metadata.hidden_size}, "
                    f"Hidden2: {metadata.hidden_size2}, "
                    f"Output: {metadata.output_size}")
        else:
            info = (f"Type: {metadata.model_type.upper()}, "
                    f"Input: {metadata.input_size}, "
                    f"Hidden: {metadata.hidden_size}, "
                    f"Output: {metadata.output_size}")
        if metadata.extra_info:
            info += f", Info: {metadata.extra_info}"
        return info
    else:
        return f"Unknown format: {filename}"



def infer_architecture_from_model(model_path: Path, device: torch.device):
    try:
        checkpoint = torch.load(model_path, map_location=device)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'q_network_state_dict' in checkpoint:
            state_dict = checkpoint['q_network_state_dict']
            if 'convLayers.0.weight' in state_dict:
                model_type = "pixel"
            else:
                model_type = "ray"
        elif isinstance(checkpoint, dict) and any(k.startswith('convLayers') for k in checkpoint.keys()):
            state_dict = checkpoint
            model_type = "pixel"
        elif isinstance(checkpoint, dict) and any(k.startswith('fc') for k in checkpoint.keys()):
            state_dict = checkpoint
            model_type = "pvp"
        else:
            state_dict = checkpoint
            model_type = "pvp"

        # Check for pixel model structure (ConvQN)
        if 'convLayers.0.weight' in state_dict:
            # head[0] is the first linear layer: (feat_dim + extra_dim) -> hidden
            if 'head.0.weight' in state_dict:
                head_input_size = state_dict['head.0.weight'].shape[1]  # feat_dim + extra_dim
                hidden_size = state_dict['head.0.weight'].shape[0]
                # head[2] is the output layer: hidden -> num_actions
                if 'head.2.weight' in state_dict:
                    output_size = state_dict['head.2.weight'].shape[0]
                    # For pixel models, input_size in metadata represents the combined input (feat_dim + extra_dim)
                    return ModelMetadata(model_type, head_input_size, hidden_size, output_size, "")
            return None

        # Check for FC model structure (Ray/PvP/Genetic)
        if 'fc1.weight' in state_dict:
            input_size = state_dict['fc1.weight'].shape[1]
            hidden_size = state_dict['fc1.weight'].shape[0]

            if 'fc3.weight' in state_dict and 'fc4.weight' not in state_dict:
                # genetic model uses 3 layers
                hidden_size2 = state_dict['fc2.weight'].shape[0]
                output_size = state_dict['fc3.weight'].shape[0]
                model_type = "genetic"
                return ModelMetadata(model_type, input_size, hidden_size, output_size, "", hidden_size2)
            elif 'fc4.weight' in state_dict:
                # ray, pixel and pvp models use 4 layers
                output_size = state_dict['fc4.weight'].shape[0]
                return ModelMetadata(model_type, input_size, hidden_size, output_size, "")
            else:
                return None
    except Exception as e:
        print(f"  Error inferring architecture: {e}")
        pass

    return None


def load_model_with_metadata(model_path: Path, metadata: Optional[ModelMetadata], device: torch.device):
    if metadata is None:
        print(f"  No metadata found, attempting to infer architecture...")
        metadata = infer_architecture_from_model(model_path, device)
        if metadata:
            if metadata.hidden_size2 is not None:
                print(f"  Inferred architecture: {metadata.input_size}->{metadata.hidden_size}->{metadata.hidden_size2}->{metadata.output_size}")
            else:
                print(f"  Inferred architecture: {metadata.input_size}->{metadata.hidden_size}->{metadata.output_size}")
        else:
            print(f"  Could not infer architecture, using defaults (17->64->3)")
            metadata = ModelMetadata("pvp", 17, 64, 3, "")

# pixel model uses ConvQN architecture
# ray, pixel and pvp models use NeuralNetwork architecture
    if metadata.model_type == "pixel":
        network = ConvQN(
            in_channels=5,
            extra_dim=8,
            num_actions=metadata.output_size,
            feat_dim=64,
            hidden=metadata.hidden_size,
            device=device
        )
    else:
        network = NeuralNetwork(
            input_size=metadata.input_size,
            hidden_size=metadata.hidden_size,
            output_size=metadata.output_size,
            device=device,
            hidden_size2=metadata.hidden_size2 if hasattr(metadata, 'hidden_size2') else None
        )

    # Load the state dict
    try:
        checkpoint = torch.load(model_path, map_location=device)

        # Extract state dict based on checkpoint format
        if isinstance(checkpoint, dict) and 'q_network_state_dict' in checkpoint:
            # Ray/Pixel model checkpoint format
            state_dict = checkpoint['q_network_state_dict']
            if metadata.model_type == "pixel":
                print(f"  Detected Pixel model checkpoint format")
            else:
                print(f"  Detected Ray model checkpoint format")
        elif isinstance(checkpoint, dict) and any(k.startswith('convLayers') for k in checkpoint.keys()):
            # Direct pixel model state dict
            state_dict = checkpoint
            print(f"  Detected Pixel model direct state dict format")
        elif isinstance(checkpoint, dict) and any(k.startswith('fc') for k in checkpoint.keys()):
            # Direct state dict (PvP format)
            state_dict = checkpoint
            print(f"  Detected direct state dict format (PvP)")
        else:
            # Try as direct state dict
            state_dict = checkpoint

        network.load_state_dict(state_dict)
        if metadata.hidden_size2 is not None:
            print(f"Loaded {metadata.model_type} model: {metadata.input_size}->{metadata.hidden_size}->{metadata.hidden_size2}->{metadata.output_size}")
        else:
            print(f"Loaded {metadata.model_type} model: {metadata.input_size}->{metadata.hidden_size}->{metadata.output_size}")
    except Exception as e:
        print(f" Warning: Could not load model weights: {e}")
        print("  Using randomly initialized network")
        import traceback
        traceback.print_exc()

    return network


def demo_single_model(model_path: Path, metadata: ModelMetadata,
                     grid_size: int = 10, render_delay: int = 10, keep_open: bool = True,
                     show_ray_lines: bool = True, record_video: bool = False, try_to_win: bool = True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"Demo: {model_path.name}")
    print(f"Architecture: {format_model_info(metadata, model_path.name)}")
    print(f"{'='*60}")

    # Load the model
    network = load_model_with_metadata(model_path, metadata, device)

    # Check if this is a pixel model
    is_pixel_model = metadata.model_type == "pixel"

    # Determine state size from model (17 for PvP, 21 for Ray)
    input_size = metadata.input_size if metadata else 17
    state_size = input_size  # Use model's input size

    # Create single-player game (NOT PvP) - similar to ray model
    game = SnakeGameSingle(grid_size=grid_size, display=True, render_delay=render_delay,
                           state_size=state_size, use_pixel_state=is_pixel_model,
                           show_ray_lines=show_ray_lines)
    game.set_network(network)

    print("\nGame started! Neural network visualization is shown on the right.")
    if try_to_win and record_video:
        print("Will automatically restart until snake wins (max 100 attempts).")
    print("Close the game window to exit.\n")

    # Check if this is a genetic model (4 actions instead of 3)
    is_genetic_model = metadata.model_type == "genetic" and metadata.output_size == 4

    # Video recording setup
    video_frames = []
    recording = False

    if record_video:
        try:
            import imageio
            recording = True
            if try_to_win:
                print("Video recording enabled. Will save video only if snake wins.")
            else:
                print("Video recording enabled. Will save video when game ends.")
        except ImportError:
            print("Warning: imageio not available. Install with 'pip install imageio' for video recording.")
            record_video = False

    import pygame

    # Outer loop for restarting games when "try to win" is enabled
    attempt = 0
    max_attempts = 1000 if try_to_win and record_video else 1

    while attempt < max_attempts:
        # Reset game for new attempt
        state = game.reset()
        step = 0
        game_finished = False

        # Clear video frames for new attempt if trying to win
        if try_to_win and record_video:
            video_frames = []
            attempt += 1
            if attempt > 1:
                print(f"\n{'='*60}")
                print(f"Attempt {attempt}/{max_attempts} - Restarting game...")
                print(f"{'='*60}\n")
        else:
            attempt = 1  # Single attempt if not trying to win

        # Inner game loop
        while True:
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    game.close()
                    return

            # Continue game logic if not finished
            if not game_finished and not game.done:
                # Get action from neural network
                action, activations = network.act(state, return_activations=True)

                # Store original action for visualization (before conversion)
                original_action = action

                # Convert genetic model actions (0-3: UP, RIGHT, DOWN, LEFT) to demo format (0-2: straight, right, left)
                if is_genetic_model:
                    # Genetic models output absolute directions, need to convert to relative
                    directions = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
                    target_direction = directions[action]
                    current_direction = game.direction

                    if target_direction == current_direction:
                        # Same direction = straight
                        action = 0
                    else:
                        # Find if it's a right or left turn
                        current_idx = Direction.get_index(current_direction)
                        target_idx = Direction.get_index(target_direction)
                        # Calculate turn direction
                        if (target_idx - current_idx) % 4 == 1:
                            action = 1  # Right turn
                        elif (target_idx - current_idx) % 4 == 3:
                            action = 2  # Left turn
                        else:
                            # 180 degree turn, treat as right turn
                            action = 1

                # Store state and activations for visualization
                # Pixel models use different visualization but still show activations
                if is_pixel_model:
                    # For pixel models, we don't need to store the input state for visualization
                    # The pixel visualizer will use the activations directly
                    game.last_state = None
                else:
                    game.last_state = state
                game.last_activations = activations
                # Store original action for genetic models (0-3), converted action for others (0-2)
                game.last_action = original_action if is_genetic_model else action

                # Step the game
                state, reward, done, info = game.step(action)
                step += 1

                if game.done and not game_finished:
                    # Show final score
                    print(f"\nGame finished!")
                    print(f"Final score: {game.score}")
                    print(f"Steps: {step}")
                    print(f"Snake length: {len(game.snake_positions)}")
                    if game.won:
                        print(f"Result: Snake won! (filled the entire grid)")
                    else:
                        # Get death reason from info dict or game object
                        death_reason = info.get('death_reason') if isinstance(info, dict) else None
                        if not death_reason and hasattr(game, 'death_reason'):
                            death_reason = game.death_reason

                        if death_reason:
                            print(f"Death reason: {death_reason}")
                        else:
                            print(f"Death reason: Unknown")

                        # Get death position from info dict
                        death_position = info.get('death_position') if isinstance(info, dict) else None
                        if death_position:
                            print(f"Death position: ({death_position[0]}, {death_position[1]})")
                        elif hasattr(game, 'snake_positions') and game.snake_positions:
                            head = game.snake_positions[0]
                            print(f"Head position at death: ({head[0]}, {head[1]})")

                        if hasattr(game, 'direction'):
                            print(f"Direction at death: {game.direction.name}")
                    game_finished = True

                    # Save video if conditions are met
                    should_save = False
                    if record_video and recording and video_frames:
                        if try_to_win:
                            # Only save if snake won
                            should_save = game.won
                        else:
                            # Save regardless of win/loss
                            should_save = True

                    if should_save:
                        try:
                            import imageio
                            import os
                            from datetime import datetime

                            # Create videos directory if it doesn't exist
                            videos_dir = Path(__file__).parent / "videos"
                            videos_dir.mkdir(exist_ok=True)

                            # Generate filename with timestamp
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            model_name = model_path.stem
                            if game.won:
                                video_filename = f"snake_win_{model_name}_{timestamp}.mp4"
                            else:
                                video_filename = f"snake_game_{model_name}_{timestamp}.mp4"
                            video_path = videos_dir / video_filename

                            print(f"\nSaving video to {video_path}...")
                            imageio.mimsave(str(video_path), video_frames, fps=10)
                            print(f"Video saved successfully! ({len(video_frames)} frames)")
                            if game.won:
                                print(f"  Snake won with score: {game.score}")
                            else:
                                print(f"  Game ended with score: {game.score}")
                        except Exception as e:
                            print(f"Error saving video: {e}")
                            import traceback
                            traceback.print_exc()

                        # If we saved the video and we're in try_to_win mode, we're done
                        if try_to_win and game.won:
                            if not keep_open:
                                game.close()
                                return
                            # If keep_open, break out of inner loop but stay in outer loop
                            break
                    elif record_video and recording and try_to_win and not game.won:
                        print(f"Video not saved (snake didn't win, score: {game.score})")
                        # Restart the game by breaking out of inner loop
                        break

                    if not try_to_win:
                        # If not trying to win, exit normally
                        if not keep_open:
                            game.close()
                            return
                        # If keep_open, break out of inner loop
                        break

            # Always render (even after game ends if keep_open is True)
            game.render()

            # Capture frame for video if recording
            # Continue recording if: game is active, or game finished and (won or not in try_to_win mode)
            should_capture = False
            if record_video and recording:
                if not game_finished:
                    should_capture = True
                elif game_finished:
                    # After game ends, capture if: won (always), or not in try_to_win mode (save all games)
                    should_capture = game.won or not try_to_win

            if should_capture:
                try:
                    # Limit frames to prevent excessive memory usage (max ~10 minutes at 10 fps)
                    if len(video_frames) < 6000:
                        # Capture the entire window
                        frame = pygame.surfarray.array3d(game.window)
                        frame = frame.swapaxes(0, 1)  # Fix axis orientation
                        # Convert to uint8 if needed and ensure RGB format
                        import numpy as np
                        if frame.dtype != np.uint8:
                            frame = frame.astype(np.uint8)
                        video_frames.append(frame)
                except Exception as e:
                    print(f"Warning: Could not capture frame: {e}")

            # If keep_open and game finished, just keep rendering until window is closed
            if keep_open and game_finished:
                # Continue rendering but don't step the game
                continue

        # After inner loop exits, check if we should restart
        if try_to_win and record_video and not game.won:
            # Continue outer loop to restart
            continue
        else:
            # Exit outer loop (either not trying to win, or we won)
            break

    # If we exhausted attempts
    if try_to_win and record_video and attempt >= max_attempts and not game.won:
        print(f"\nReached maximum attempts ({max_attempts}). No winning game found.")
        if not keep_open:
            game.close()
            return


def simulate_and_save_win(model_path: Path, metadata: ModelMetadata,
                         grid_size: int = 10, max_games: int = 1000,
                         state_size: int = None, use_pixel_state: bool = False,
                         output_dir: Path = None):
    #Simulate games without rendering and save the first winning game.

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"Simulating games to find a win: {model_path.name}")
    print(f"Architecture: {format_model_info(metadata, model_path.name)}")
    print(f"Max games: {max_games}")
    print(f"{'='*60}\n")

    # Load the model
    network = load_model_with_metadata(model_path, metadata, device)

    # Determine state size
    if state_size is None:
        state_size = metadata.input_size if metadata else 17

    # Check if this is a pixel model
    is_pixel_model = metadata.model_type == "pixel" if metadata else False
    if use_pixel_state is None:
        use_pixel_state = is_pixel_model

    # Check if this is a genetic model (4 actions instead of 3)
    is_genetic_model = metadata.model_type == "genetic" and metadata.output_size == 4

    # Create game without display
    game = SnakeGameSingle(grid_size=grid_size, display=False, render_delay=0,
                          state_size=state_size, use_pixel_state=use_pixel_state,
                          show_ray_lines=False)
    game.set_network(network)

    # Setup output directory
    if output_dir is None:
        output_dir = Path(__file__).parent / "replays"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Simulate games
    highest_score = 0
    for game_num in range(1, max_games + 1):
        # Reset game
        state = game.reset()

        # Record initial state
        initial_state = {
            'snake_positions': list(game.snake_positions),
            'direction': game.direction.name,
            'food_position': game.food_position,
            'grid_size': grid_size,
            'score': 0,
            'steps': 0
        }

        # Record game actions and states
        actions = []
        states_history = []
        food_positions = []  # Track food positions at each step to ensure deterministic replay

        # Play game
        step = 0
        while not game.done:
            # Save current food position before step
            food_positions.append(game.food_position)

            # Get action from neural network
            action = network.act(state, return_activations=False)

            # Store original action
            original_action = action

            # Convert genetic model actions if needed
            if is_genetic_model:
                directions = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
                target_direction = directions[action]
                current_direction = game.direction

                if target_direction == current_direction:
                    action = 0  # Straight
                else:
                    current_idx = Direction.get_index(current_direction)
                    target_idx = Direction.get_index(target_direction)
                    if (target_idx - current_idx) % 4 == 1:
                        action = 1  # Right turn
                    elif (target_idx - current_idx) % 4 == 3:
                        action = 2  # Left turn
                    else:
                        action = 1  # 180 degree turn, treat as right

            # Store action (save original for genetic models)
            actions.append(original_action if is_genetic_model else action)

            # Store state snapshot (for pixel models, store simplified state)
            if use_pixel_state:
                # For pixel models, store the game state in a simpler format
                states_history.append({
                    'snake_positions': list(game.snake_positions),
                    'direction': game.direction.name,
                    'food_position': game.food_position,
                    'score': game.score
                })
            else:
                # For regular models, store the state vector
                if isinstance(state, np.ndarray):
                    states_history.append(state.tolist())
                else:
                    states_history.append(None)

            state, reward, done, info = game.step(action)
            step += 1


        # Track highest score
        if game.score > highest_score:
            highest_score = game.score

        # Display progress every 100 games
        if game_num % 20 == 0:
            print(f"Simulated {game_num} games... Highest score: {highest_score}")

        # Check if won
        if game.won:
            print(f"\nFound winning game after {game_num} simulations!")
            print(f"  Score: {game.score}")
            print(f"  Steps: {step}")
            print(f"  Snake length: {len(game.snake_positions)}")

            # Create replay data
            replay_data = {
                'initial_state': initial_state,
                'actions': actions,
                'food_positions': food_positions,  # Save food position sequence for deterministic replay
                'states_history': states_history,
                'final_score': game.score,
                'final_steps': step,
                'won': True,
                'model_name': model_path.name,
                'model_metadata': {
                    'model_type': metadata.model_type,
                    'input_size': metadata.input_size,
                    'hidden_size': metadata.hidden_size,
                    'output_size': metadata.output_size,
                    'hidden_size2': getattr(metadata, 'hidden_size2', None)
                },
                'grid_size': grid_size,
                'is_genetic_model': is_genetic_model,
                'use_pixel_state': use_pixel_state,
                'timestamp': datetime.now().isoformat()
            }

            # Save replay
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = model_path.stem
            replay_filename = f"replay_win_{model_name}_{timestamp}.pkl"
            replay_path = output_dir / replay_filename

            with open(replay_path, 'wb') as f:
                pickle.dump(replay_data, f)

            print(f"Replay saved to: {replay_path}")
            return replay_path

    print(f"\nNo winning game found after {max_games} simulations.")
    return None


def replay_game(replay_path: Path, render_delay: int = 10, show_ray_lines: bool = True):
    #Replay a saved game with rendering.

    print(f"\n{'='*60}")
    print(f"Replaying game: {replay_path.name}")
    print(f"{'='*60}\n")

    # Load replay data
    with open(replay_path, 'rb') as f:
        replay_data = pickle.load(f)

    grid_size = replay_data['grid_size']
    initial_state = replay_data['initial_state']
    actions = replay_data['actions']
    food_positions = replay_data.get('food_positions', [])  # Get saved food positions
    is_genetic_model = replay_data.get('is_genetic_model', False)
    use_pixel_state = replay_data.get('use_pixel_state', False)

    # Determine state size from metadata
    metadata = replay_data.get('model_metadata', {})
    state_size = metadata.get('input_size', 17)

    # Create game with display
    game = SnakeGameSingle(grid_size=grid_size, display=True, render_delay=render_delay,
                          state_size=state_size, use_pixel_state=use_pixel_state,
                          show_ray_lines=show_ray_lines)

    # Restore initial state
    game.snake_positions = deque(initial_state['snake_positions'])
    game.direction = Direction[initial_state['direction']]
    game.food_position = initial_state['food_position']
    game.score = initial_state['score']
    game.steps = initial_state['steps']
    game.alive = True
    game.done = False
    game.won = False
    game.last_food_step = 0
    game.distance_to_food = abs(game.snake_positions[0][0] - game.food_position[0]) + abs(game.snake_positions[0][1] - game.food_position[1]) if game.food_position else 0

    print(f"Replaying {len(actions)} actions...")
    print(f"Final score: {replay_data['final_score']}")
    print(f"Final steps: {replay_data['final_steps']}")
    print("Close the game window to exit.\n")

    import pygame

    # Replay actions
    action_idx = 0
    while action_idx < len(actions) and not game.done:
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game.close()
                return

        # Set food position from saved sequence before step (ensures deterministic replay)
        if action_idx < len(food_positions):
            game.food_position = food_positions[action_idx]

        # Get action
        original_action = actions[action_idx]

        # Convert genetic model actions if needed
        if is_genetic_model:
            directions = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
            target_direction = directions[original_action]
            current_direction = game.direction

            if target_direction == current_direction:
                action = 0  # Straight
            else:
                current_idx = Direction.get_index(current_direction)
                target_idx = Direction.get_index(target_direction)
                if (target_idx - current_idx) % 4 == 1:
                    action = 1  # Right turn
                elif (target_idx - current_idx) % 4 == 3:
                    action = 2  # Left turn
                else:
                    action = 1  # 180 degree turn
        else:
            action = original_action

        # Step the game
        state, reward, done, info = game.step(action)

        # After step, override the food position with the next saved position
        # This ensures deterministic replay by overriding random food placement
        if action_idx + 1 < len(food_positions):
            game.food_position = food_positions[action_idx + 1]
            # Update distance to food
            if game.food_position:
                game.distance_to_food = abs(game.snake_positions[0][0] - game.food_position[0]) + abs(game.snake_positions[0][1] - game.food_position[1])
            else:
                game.distance_to_food = 0

        action_idx += 1

        # Render
        game.render()

    # Keep window open after replay finishes
    print("\nReplay finished! Close the window to exit.")
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game.close()
                return
        game.render()


def demo_pvp_models(model_paths: list, metadata_list: list,
                   grid_size: int = 20, num_snakes: int = None,
                   render_delay: int = 10, keep_open: bool = True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if num_snakes is None:
        num_snakes = min(len(model_paths), 4)

    print(f"\n{'='*60}")
    print(f"PvP Demo with {num_snakes} models")
    for i, (path, meta) in enumerate(zip(model_paths[:num_snakes], metadata_list[:num_snakes])):
        print(f"  Snake {i}: {path.name} - {format_model_info(meta, path.name)}")
    print(f"{'='*60}")

    # Load models
    networks = {}
    for i, (model_path, metadata) in enumerate(zip(model_paths[:num_snakes], metadata_list[:num_snakes])):
        network = load_model_with_metadata(model_path, metadata, device)
        networks[i] = network

    # Create game
    game = SnakeGamePvP(grid_size=grid_size, num_snakes=num_snakes, display=True, render_delay=render_delay)
    states = game.reset()
    # Set networks AFTER reset so they're applied to the newly created snakes
    game.set_snake_networks(networks)

    print("\nGame started! Click on snake names to view their neural networks.")
    print("Close the game window to exit.\n")

    # Initial render to show the game immediately
    game.render()

    # Small delay to ensure window is ready
    import time
    time.sleep(0.1)

    # Game loop
    step = 0
    game_finished = False

    import pygame

    while True:
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game.close()
                return
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    mouse_x, mouse_y = event.pos
                    game.handle_mouse_click(mouse_x, mouse_y)

        # Continue game logic if not finished
        if not game_finished and not game.done:
            # Get actions from neural networks
            actions = {}
            if states:  # Only process if there are alive snakes
                for snake_id, state in states.items():
                    if snake_id in networks:
                        try:
                            action = networks[snake_id].act(state)
                            actions[snake_id] = action
                        except Exception as e:
                            print(f"Error getting action for snake {snake_id}: {e}")
                            actions[snake_id] = 0
                    else:
                        actions[snake_id] = 0
            else:
                # No alive snakes, game should be done
                game.done = True

            # Step the game
            if not game.done:
                try:
                    states, rewards, done, info = game.step(actions)
                    step += 1
                except Exception as e:
                    print(f"Error in game step: {e}")
                    import traceback
                    traceback.print_exc()
                    break

            if game.done and not game_finished:
                # Show final results
                snake_scores = game.get_snake_scores()
                print(f"\nGame finished!")
                print(f"Scores: {snake_scores}")
                if game.winner is not None:
                    print(f"Winner: Snake {game.winner}")
                print(f"Steps: {step}")
                game_finished = True
                if not keep_open:
                    game.close()
                    return

        # Always render (even after game ends if keep_open is True)
        game.render()

        # If keep_open and game finished, just keep rendering until window is closed
        if keep_open and game_finished:
            # Continue rendering but don't step the game
            continue


def migrate_model(source_path: Path, target_dir: Path, extra_info: str = ""):
    # Migrate a model file to the new naming format (utility function)
    # Args:
    #   source_path: Path to source model file
    #   target_dir: Directory to save migrated model
    #   extra_info: Extra info to include in filename (e.g., 'gen30', 'final')
    print(f"\nMigrating: {source_path.name}")

    # Check if already in correct format
    existing_meta = parse_model_filename(source_path.name)
    if existing_meta:
        print(f"  Already in correct format")
        if target_dir != source_path.parent:
            target_path = target_dir / source_path.name
            import shutil
            shutil.copy2(source_path, target_path)
            print(f"  Copied to: {target_path}")
        return

    # Extract architecture
    print(f"  Extracting architecture...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arch = infer_architecture_from_model(source_path, device)

    if not arch:
        print(f"  Could not determine architecture, using defaults (pvp_17_64_3)")
        arch = ModelMetadata("pvp", 17, 64, 3, "")

    print(f"  Architecture: {arch.input_size}->{arch.hidden_size}->{arch.output_size}")

    # Extract extra info from filename if not provided
    if not extra_info:
        stem = source_path.stem.lower()
        if 'gen' in stem:
            gen_match = re.search(r'gen[_\s]*(\d+)', stem)
            if gen_match:
                extra_info = f"gen{gen_match.group(1)}"
        elif 'final' in stem:
            extra_info = "final"
        elif 'episode' in stem:
            ep_match = re.search(r'episode[_\s]*(\d+)', stem)
            if ep_match:
                extra_info = f"ep{ep_match.group(1)}"

    # Create new filename
    new_filename = create_model_filename(
        model_type=arch.model_type,
        input_size=arch.input_size,
        hidden_size=arch.hidden_size,
        output_size=arch.output_size,
        extra_info=extra_info
    )

    target_path = target_dir / new_filename

    # Copy file
    if target_path.exists():
        response = input(f"  File exists: {target_path.name}. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print(f"  Skipped")
            return

    import shutil
    shutil.copy2(source_path, target_path)
    print(f"  Migrated to: {target_path.name}")


class DemoGUI:
    def __init__(self):
        if tk is None:
            print("tkinter not available. Please install tkinter for GUI support.")
            return

        self.root = tk.Tk()
        self.root.title("Snake AI Model Demo")
        self.root.geometry("650x750")

        # Models directory
        self.models_dir = Path(__file__).parent / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Replays directory
        self.replays_dir = Path(__file__).parent / "replays"
        self.replays_dir.mkdir(parents=True, exist_ok=True)

        # Model data
        self.model_list = []
        self.model_paths = []

        # Create GUI
        self.create_widgets()
        self.scan_models()

    def create_widgets(self):
        # Title
        title_label = tk.Label(self.root, text="Snake AI Model Demo",
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=10)

        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Create Single Model tab
        self.single_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.single_tab, text="Single Model")
        self.create_single_tab()

        # Create PvP tab
        self.pvp_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.pvp_tab, text="PvP")
        self.create_pvp_tab()

        # Status/Log section (shared across tabs)
        log_frame = ttk.LabelFrame(self.root, text="Status", padding=5)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.log_text = scrolledtext.ScrolledText(log_frame, height=8, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log("Ready. Select models and click Start to begin.")

    def create_single_tab(self):
        # Models section
        models_frame = ttk.LabelFrame(self.single_tab, text="Available Models", padding=10)
        models_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Refresh button
        refresh_btn = tk.Button(models_frame, text="Refresh Models",
                               command=self.scan_models)
        refresh_btn.pack(anchor=tk.W, pady=5)

        # Single model dropdown
        single_frame = tk.Frame(models_frame)
        single_frame.pack(fill=tk.X, pady=5)
        tk.Label(single_frame, text="Select Model:").pack(side=tk.LEFT, padx=5)
        self.single_model_var = tk.StringVar()
        self.single_model_combo = ttk.Combobox(single_frame, textvariable=self.single_model_var,
                                                state="readonly", width=50)
        self.single_model_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.single_model_combo.bind('<<ComboboxSelected>>', self.on_single_model_select)

        # Model info label
        self.single_model_info_label = tk.Label(models_frame, text="Select a model to see details",
                                         wraplength=550, justify=tk.LEFT)
        self.single_model_info_label.pack(fill=tk.X, pady=5)

        # Settings section
        settings_frame = ttk.LabelFrame(self.single_tab, text="Game Settings", padding=10)
        settings_frame.pack(fill=tk.X, padx=10, pady=5)

        # Grid size
        grid_frame = tk.Frame(settings_frame)
        grid_frame.pack(fill=tk.X, pady=2)
        tk.Label(grid_frame, text="Grid Size:").pack(side=tk.LEFT)
        self.single_grid_size_var = tk.StringVar(value="10")
        grid_spinbox = tk.Spinbox(grid_frame, from_=10, to=30, textvariable=self.single_grid_size_var, width=10)
        grid_spinbox.pack(side=tk.LEFT, padx=5)

        # Render delay/FPS
        fps_frame = tk.Frame(settings_frame)
        fps_frame.pack(fill=tk.X, pady=2)
        tk.Label(fps_frame, text="FPS (0=max speed):").pack(side=tk.LEFT)
        self.single_fps_var = tk.StringVar(value="10")
        fps_spinbox = tk.Spinbox(fps_frame, from_=0, to=60, textvariable=self.single_fps_var, width=10)
        fps_spinbox.pack(side=tk.LEFT, padx=5)

        # Show ray lines option (for ray models)
        ray_frame = tk.Frame(settings_frame)
        ray_frame.pack(fill=tk.X, pady=2)
        self.single_show_ray_lines_var = tk.BooleanVar(value=False)
        ray_checkbox = tk.Checkbutton(ray_frame, text="Show Ray Lines (for ray models)",
                                      variable=self.single_show_ray_lines_var)
        ray_checkbox.pack(side=tk.LEFT)

        # Record video option
        video_frame = tk.Frame(settings_frame)
        video_frame.pack(fill=tk.X, pady=2)
        self.single_record_video_var = tk.BooleanVar(value=False)
        video_checkbox = tk.Checkbutton(video_frame, text="Record Video",
                                       variable=self.single_record_video_var,
                                       command=self.update_video_options)
        video_checkbox.pack(side=tk.LEFT, padx=5)

        # Try to win option (only enabled when recording)
        self.single_try_to_win_var = tk.BooleanVar(value=True)
        self.single_try_to_win_checkbox = tk.Checkbutton(video_frame, text="Try to Win (only save if snake wins)",
                                                  variable=self.single_try_to_win_var,
                                                  state=tk.DISABLED)
        self.single_try_to_win_checkbox.pack(side=tk.LEFT, padx=5)

        # Simulation mode option
        sim_frame = tk.Frame(settings_frame)
        sim_frame.pack(fill=tk.X, pady=2)
        self.single_simulate_mode_var = tk.BooleanVar(value=False)
        sim_checkbox = tk.Checkbutton(sim_frame, text="Simulation Mode",
                                     variable=self.single_simulate_mode_var,
                                     command=self.update_simulation_options)
        sim_checkbox.pack(side=tk.LEFT, padx=5)

        # Max games for simulation
        self.single_max_games_var = tk.StringVar(value="1000")
        self.single_max_games_label = tk.Label(sim_frame, text="Max games:", state=tk.DISABLED)
        self.single_max_games_label.pack(side=tk.LEFT, padx=5)
        self.single_max_games_spinbox = tk.Spinbox(sim_frame, from_=1, to=10000, textvariable=self.single_max_games_var,
                                            width=10, state=tk.DISABLED)
        self.single_max_games_spinbox.pack(side=tk.LEFT, padx=5)

        # Buttons section
        buttons_frame = tk.Frame(self.single_tab)
        buttons_frame.pack(fill=tk.X, padx=10, pady=10)

        self.start_single_btn = tk.Button(buttons_frame, text="Start Single Model Demo",
                                         command=self.start_single_demo,
                                         bg="#4CAF50", fg="white", font=("Arial", 10, "bold"),
                                         state=tk.DISABLED)
        self.start_single_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        self.replay_btn = tk.Button(buttons_frame, text="Replay Saved Game",
                                    command=self.replay_saved_game,
                                    bg="#FF9800", fg="white", font=("Arial", 10, "bold"))
        self.replay_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

    def create_pvp_tab(self):
        # Models section
        models_frame = ttk.LabelFrame(self.pvp_tab, text="Available Models", padding=10)
        models_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Refresh button
        refresh_btn = tk.Button(models_frame, text="Refresh Models",
                               command=self.scan_models)
        refresh_btn.pack(anchor=tk.W, pady=5)

        # PvP models dropdowns (for multi-select, up to 4 snakes)
        pvp_frame = tk.Frame(models_frame)
        pvp_frame.pack(fill=tk.X, pady=5)
        tk.Label(pvp_frame, text="PvP Models (select 2-4):").pack(anchor=tk.W, pady=2)

        self.pvp_model_vars = []
        self.pvp_model_combos = []
        for i in range(4):
            pvp_row = tk.Frame(pvp_frame)
            pvp_row.pack(fill=tk.X, pady=2)
            tk.Label(pvp_row, text=f"Snake {i+1}:").pack(side=tk.LEFT, padx=5)
            pvp_var = tk.StringVar()
            pvp_combo = ttk.Combobox(pvp_row, textvariable=pvp_var,
                                     state="readonly", width=45)
            pvp_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            pvp_combo.bind('<<ComboboxSelected>>', lambda e, idx=i: self.on_pvp_model_select(idx))
            self.pvp_model_vars.append(pvp_var)
            self.pvp_model_combos.append(pvp_combo)

        # Model info label
        self.pvp_model_info_label = tk.Label(models_frame, text="Select models to see details",
                                         wraplength=550, justify=tk.LEFT)
        self.pvp_model_info_label.pack(fill=tk.X, pady=5)

        # Settings section
        settings_frame = ttk.LabelFrame(self.pvp_tab, text="Game Settings", padding=10)
        settings_frame.pack(fill=tk.X, padx=10, pady=5)

        # Grid size
        grid_frame = tk.Frame(settings_frame)
        grid_frame.pack(fill=tk.X, pady=2)
        tk.Label(grid_frame, text="Grid Size:").pack(side=tk.LEFT)
        self.pvp_grid_size_var = tk.StringVar(value="20")
        grid_spinbox = tk.Spinbox(grid_frame, from_=10, to=30, textvariable=self.pvp_grid_size_var, width=10)
        grid_spinbox.pack(side=tk.LEFT, padx=5)

        # Render delay/FPS
        fps_frame = tk.Frame(settings_frame)
        fps_frame.pack(fill=tk.X, pady=2)
        tk.Label(fps_frame, text="FPS (0=max speed):").pack(side=tk.LEFT)
        self.pvp_fps_var = tk.StringVar(value="10")
        fps_spinbox = tk.Spinbox(fps_frame, from_=0, to=60, textvariable=self.pvp_fps_var, width=10)
        fps_spinbox.pack(side=tk.LEFT, padx=5)

        # Buttons section
        buttons_frame = tk.Frame(self.pvp_tab)
        buttons_frame.pack(fill=tk.X, padx=10, pady=10)

        self.start_pvp_btn = tk.Button(buttons_frame, text="Start PvP Demo",
                                       command=self.start_pvp_demo,
                                       bg="#2196F3", fg="white", font=("Arial", 10, "bold"),
                                       state=tk.DISABLED)
        self.start_pvp_btn.pack(fill=tk.X, expand=True)

    def log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def update_video_options(self):
        # Enable/disable "Try to Win" checkbox based on "Record Video" state
        if hasattr(self, 'single_try_to_win_checkbox'):
            if self.single_record_video_var.get():
                self.single_try_to_win_checkbox.config(state=tk.NORMAL)
            else:
                self.single_try_to_win_checkbox.config(state=tk.DISABLED)

    def update_simulation_options(self):
        # Enable/disable simulation options based on "Simulation Mode" state
        if hasattr(self, 'single_max_games_label') and hasattr(self, 'single_max_games_spinbox'):
            if self.single_simulate_mode_var.get():
                self.single_max_games_label.config(state=tk.NORMAL)
                self.single_max_games_spinbox.config(state=tk.NORMAL)
            else:
                self.single_max_games_label.config(state=tk.DISABLED)
                self.single_max_games_spinbox.config(state=tk.DISABLED)

    def scan_models(self):
        self.log("Scanning models directory...")
        self.single_model_combo['values'] = []
        self.model_list = []
        self.model_paths = []

        if not self.models_dir.exists():
            self.log(f"Models directory not found: {self.models_dir}")
            return

        models = list_models_in_directory(self.models_dir)

        if not models:
            self.log(f"No model files found in: {self.models_dir}")
            return

        model_items = list(models.items())
        display_names = []
        for filename, metadata in model_items:
            display_name = filename
            if metadata:
                if metadata.hidden_size2 is not None:
                    display_name += f" ({metadata.model_type.upper()}, {metadata.input_size}->{metadata.hidden_size}->{metadata.hidden_size2}->{metadata.output_size})"
                else:
                    display_name += f" ({metadata.model_type.upper()}, {metadata.input_size}->{metadata.hidden_size}->{metadata.output_size})"
            else:
                display_name += " (Unknown format)"

            display_names.append(display_name)
            self.model_list.append((filename, metadata))
            self.model_paths.append(self.models_dir / filename)

        # Update all comboboxes with same values
        self.single_model_combo['values'] = display_names
        for combo in self.pvp_model_combos:
            combo['values'] = display_names

        if display_names:
            self.single_model_combo.current(0)
            self.on_single_model_select(None)

        self.log(f"Found {len(models)} model(s)")
        self.update_button_states()

    def on_single_model_select(self, event):
        selection = self.single_model_var.get()
        if selection:
            try:
                idx = self.single_model_combo['values'].index(selection)
                filename, metadata = self.model_list[idx]
                if metadata:
                    info = format_model_info(metadata, filename)
                else:
                    info = f"Unknown format: {filename} (will attempt to infer architecture)"
                self.single_model_info_label.config(text=info)
            except (ValueError, IndexError):
                pass
        self.update_button_states()

    def on_pvp_model_select(self, combo_idx):
        var = self.pvp_model_vars[combo_idx]
        selection = var.get()
        if selection:
            try:
                idx = self.pvp_model_combos[combo_idx]['values'].index(selection)
                filename, metadata = self.model_list[idx]
                if metadata:
                    info = format_model_info(metadata, filename)
                else:
                    info = f"Unknown format: {filename} (will attempt to infer architecture)"
                self.pvp_model_info_label.config(text=info)
            except (ValueError, IndexError):
                pass
        self.update_button_states()

    def update_button_states(self):
        single_selected = bool(self.single_model_var.get())

        # Count how many PvP dropdowns have selections
        pvp_count = sum(1 for var in self.pvp_model_vars if var.get())

        self.start_single_btn.config(state=tk.NORMAL if single_selected else tk.DISABLED)
        self.start_pvp_btn.config(state=tk.NORMAL if pvp_count >= 2 else tk.DISABLED)

    def start_single_demo(self):
        selection = self.single_model_var.get()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a model first.")
            return

        try:
            idx = self.single_model_combo['values'].index(selection)
        except (ValueError, IndexError):
            messagebox.showerror("Error", "Invalid model selection.")
            return

        filename, metadata = self.model_list[idx]
        model_path = self.model_paths[idx]

        # Get settings
        try:
            grid_size = int(self.single_grid_size_var.get())
            fps = int(self.single_fps_var.get())
            render_delay = fps if fps > 0 else 0
        except ValueError:
            messagebox.showerror("Invalid Settings", "Please enter valid numbers for settings.")
            return

        # If no metadata, try to infer it
        if metadata is None:
            self.log(f"Inferring architecture for {filename}...")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            inferred_metadata = infer_architecture_from_model(model_path, device)
            if inferred_metadata:
                metadata = inferred_metadata
                self.log(f"Inferred: {format_model_info(metadata, filename)}")
            else:
                self.log(f"Could not infer architecture, using defaults")
                metadata = ModelMetadata("pvp", 17, 64, 3, "")

        self.log(f"Starting single model demo: {filename}")
        self.log("Game window opened.")
        print(f"Starting single model demo with {filename}, num_snakes=1")

        # Get show ray lines option
        show_ray_lines = self.single_show_ray_lines_var.get()

        # Get record video option
        record_video = self.single_record_video_var.get()

        # Get try to win option
        try_to_win = self.single_try_to_win_var.get()

        # Get simulation mode option
        simulate_mode = self.single_simulate_mode_var.get()

        # If simulation mode, run simulation instead of demo
        if simulate_mode:
            try:
                max_games = int(self.single_max_games_var.get())
            except ValueError:
                messagebox.showerror("Invalid Settings", "Please enter a valid number for max games.")
                return

            self.log(f"Starting simulation mode: {filename}")
            self.log(f"Will simulate up to {max_games} games to find a win...")

            # Run simulation in separate thread
            def run_simulation():
                replay_path = simulate_and_save_win(
                    model_path, metadata, grid_size, max_games,
                    state_size=None, use_pixel_state=None, output_dir=None
                )
                if replay_path:
                    self.log(f"Winning game found and saved: {replay_path.name}")
                    self.log("You can replay it using the 'Replay Game' option.")
                else:
                    self.log(f"No winning game found after {max_games} simulations.")

            thread = threading.Thread(target=run_simulation, daemon=True)
            thread.start()
        else:
            # Run in separate thread
            thread = threading.Thread(target=demo_single_model,
                                     args=(model_path, metadata, grid_size, render_delay, True, show_ray_lines, record_video, try_to_win),
                                     daemon=True)
            thread.start()

    def start_pvp_demo(self):
        # Get selected models from dropdowns
        selected_indices = []
        for i, var in enumerate(self.pvp_model_vars):
            selection = var.get()
            if selection:
                try:
                    idx = self.pvp_model_combos[i]['values'].index(selection)
                    selected_indices.append((i, idx))
                except (ValueError, IndexError):
                    pass

        if len(selected_indices) < 2:
            messagebox.showwarning("Invalid Selection", "Please select at least 2 models for PvP.")
            return

        # Get selected models
        selected_models = []
        selected_metadata = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for combo_idx, list_idx in selected_indices[:4]:  # Max 4 snakes
            filename, metadata = self.model_list[list_idx]
            model_path = self.model_paths[list_idx]

            # If no metadata, try to infer it
            if metadata is None:
                inferred_metadata = infer_architecture_from_model(model_path, device)
                if inferred_metadata:
                    metadata = inferred_metadata
                else:
                    metadata = ModelMetadata("pvp", 17, 64, 3, "")

            selected_models.append(model_path)
            selected_metadata.append(metadata)

        num_snakes = len(selected_models)
        if len(selected_indices) > 4:
            self.log(f"Warning: Only using first 4 models")

        # Get settings
        try:
            grid_size = int(self.pvp_grid_size_var.get())
            fps = int(self.pvp_fps_var.get())
            # Ensure minimum delay to prevent freezing (at least 1 FPS = delay of 1)
            render_delay = fps if fps > 0 else 1
        except ValueError:
            messagebox.showerror("Invalid Settings", "Please enter valid numbers for settings.")
            return

        self.log(f"Starting PvP demo with {num_snakes} models")
        self.log("Game window will open. Close it when done.")

        # Run in separate thread
        thread = threading.Thread(target=demo_pvp_models,
                                 args=(selected_models, selected_metadata, grid_size, num_snakes, render_delay, True),
                                 daemon=True)
        thread.start()

    def replay_saved_game(self):
        """Open file dialog to select and replay a saved game."""
        from tkinter import filedialog

        replay_path = filedialog.askopenfilename(
            title="Select Replay File",
            initialdir=str(self.replays_dir),
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )

        if not replay_path:
            return

        replay_path = Path(replay_path)

        if not replay_path.exists():
            messagebox.showerror("Error", f"Replay file not found: {replay_path}")
            return

        # Get settings
        try:
            fps = int(self.single_fps_var.get())
            render_delay = fps if fps > 0 else 0
        except ValueError:
            messagebox.showerror("Invalid Settings", "Please enter valid numbers for settings.")
            return

        show_ray_lines = self.single_show_ray_lines_var.get()

        self.log(f"Replaying game: {replay_path.name}")
        self.log("Game window will open. Close it when done.")

        # Run in separate thread
        thread = threading.Thread(target=replay_game,
                                 args=(replay_path, render_delay, show_ray_lines),
                                 daemon=True)
        thread.start()

    def run(self):
        self.root.mainloop()


def main():
    if tk is None:
        print("tkinter not available. Falling back to command-line interface.")
        # Could add command-line fallback here if needed
        return

    app = DemoGUI()
    app.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)