import sys
import os
import torch
import numpy as np
import re
import threading
from pathlib import Path
from typing import Optional, Dict

# All imports are local to demo folder - self-contained
from snake_game_single import SnakeGameSingle, Direction
from snake_game_pvp import SnakeGamePvP
from neural_network import NeuralNetwork, ConvQN

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext


class ModelMetadata:

    def __init__(self, model_type: str, input_size: int, hidden_size: int,
                 output_size: int, extra_info: str = "", hidden_size2: Optional[int] = None):
        self.model_type = model_type  # e.g., 'pvp', 'ray', 'lstm', 'pixel', 'genetic'
        self.input_size = input_size
        self.hidden_size = hidden_size  # First hidden layer size
        self.hidden_size2 = hidden_size2  # Second hidden layer size (for genetic models)
        self.output_size = output_size
        self.extra_info = extra_info  # e.g., 'gen30', 'final', etc.

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
        hidden_size = int(match.group(2))  # hidden_size1
        hidden_size2 = int(match.group(3))  # hidden_size2
        output_size = int(match.group(4))
        extra_info = match.group(5) or ""
        return ModelMetadata(model_type, input_size, hidden_size, output_size, extra_info, hidden_size2)

    # Pattern: type_input_hidden_output_optional_extra (standard format)
    pattern = r'^([a-z]+)_(\d+)_(\d+)_(\d+)(?:_(.+))?$'
    match = re.match(pattern, name)

    if match:
        model_type = match.group(1)
        input_size = int(match.group(2))
        hidden_size = int(match.group(3))
        output_size = int(match.group(4))
        extra_info = match.group(5) or ""

        return ModelMetadata(model_type, input_size, hidden_size, output_size, extra_info)

    # Try to parse legacy filenames (e.g., pvp_snake_gen_30.pth)
    legacy_pattern = r'^([a-z_]+)_gen_(\d+)$'
    match = re.match(legacy_pattern, name)
    if match:
        # Default to PvP architecture
        model_type = "pvp"
        input_size = 17
        hidden_size = 64
        output_size = 3
        extra_info = f"gen{match.group(2)}"
        return ModelMetadata(model_type, input_size, hidden_size, output_size, extra_info)

    # Try final pattern
    if "final" in name.lower():
        # Default to PvP architecture
        model_type = "pvp"
        input_size = 17
        hidden_size = 64
        output_size = 3
        extra_info = "final"
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
            # Store with None metadata for files that don't match pattern
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


# ============================================================================
# Demo Functions
# ============================================================================


def infer_architecture_from_model(model_path: Path, device: torch.device):
    try:
        checkpoint = torch.load(model_path, map_location=device)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'q_network_state_dict' in checkpoint:
            state_dict = checkpoint['q_network_state_dict']
            # Check if it's a pixel model (has convLayers) or ray model (has fc1)
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
            # Pixel model: ConvQN architecture
            # Extract dimensions from state dict
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

            # Check if it's a genetic model (has fc1, fc2, fc3 but no fc4)
            if 'fc3.weight' in state_dict and 'fc4.weight' not in state_dict:
                # Genetic model: (input_size, hidden_size1, hidden_size2, output_size)
                hidden_size2 = state_dict['fc2.weight'].shape[0]
                output_size = state_dict['fc3.weight'].shape[0]
                model_type = "genetic"
                return ModelMetadata(model_type, input_size, hidden_size, output_size, "", hidden_size2)
            elif 'fc4.weight' in state_dict:
                # Standard 4-layer model (Ray/PvP)
                output_size = state_dict['fc4.weight'].shape[0]
                return ModelMetadata(model_type, input_size, hidden_size, output_size, "")
            else:
                return None
    except Exception as e:
        print(f"  Error inferring architecture: {e}")
        pass

    return None


def load_model_with_metadata(model_path: Path, metadata: Optional[ModelMetadata], device: torch.device):
    # If no metadata, try to infer from model file
    if metadata is None:
        print(f"  No metadata found, attempting to infer architecture...")
        metadata = infer_architecture_from_model(model_path, device)
        if metadata:
            if metadata.hidden_size2 is not None:
                print(f"  ✓ Inferred architecture: {metadata.input_size}→{metadata.hidden_size}→{metadata.hidden_size2}→{metadata.output_size}")
            else:
                print(f"  ✓ Inferred architecture: {metadata.input_size}→{metadata.hidden_size}→{metadata.output_size}")
        else:
            print(f"  ⚠ Could not infer architecture, using defaults (17→64→3)")
            metadata = ModelMetadata("pvp", 17, 64, 3, "")

    # Create the appropriate network architecture based on model type
    if metadata.model_type == "pixel":
        # Pixel model uses ConvQN architecture
        # Default parameters for pixel model: in_channels=5, extra_dim=8, feat_dim=64, hidden=128
        network = ConvQN(
            in_channels=5,
            extra_dim=8,
            num_actions=metadata.output_size,
            feat_dim=64,
            hidden=metadata.hidden_size,
            device=device
        )
    else:
        # Ray, PvP, and Genetic models use NeuralNetwork
        # Pass hidden_size2 for genetic models
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
            print(f"✓ Loaded {metadata.model_type} model: {metadata.input_size}→{metadata.hidden_size}→{metadata.hidden_size2}→{metadata.output_size}")
        else:
            print(f"✓ Loaded {metadata.model_type} model: {metadata.input_size}→{metadata.hidden_size}→{metadata.output_size}")
    except Exception as e:
        print(f"⚠ Warning: Could not load model weights: {e}")
        print("  Using randomly initialized network")
        import traceback
        traceback.print_exc()

    return network


def demo_single_model(model_path: Path, metadata: ModelMetadata,
                     grid_size: int = 10, render_delay: int = 10, keep_open: bool = True,
                     show_ray_lines: bool = True):
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

    # Get initial state (pixel format or vector format)
    state = game.reset()

    print("\nGame started! Neural network visualization is shown on the right.")
    print("Close the game window to exit.\n")

    # Check if this is a genetic model (4 actions instead of 3)
    is_genetic_model = metadata.model_type == "genetic" and metadata.output_size == 4

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
        print(f"  ✓ Already in correct format")
        if target_dir != source_path.parent:
            target_path = target_dir / source_path.name
            import shutil
            shutil.copy2(source_path, target_path)
            print(f"  ✓ Copied to: {target_path}")
        return

    # Extract architecture
    print(f"  Extracting architecture...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arch = infer_architecture_from_model(source_path, device)

    if not arch:
        print(f"  ⚠ Could not determine architecture, using defaults (pvp_17_64_3)")
        arch = ModelMetadata("pvp", 17, 64, 3, "")

    print(f"  Architecture: {arch.input_size}→{arch.hidden_size}→{arch.output_size}")

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
    print(f"  ✓ Migrated to: {target_path.name}")


class DemoGUI:
    def __init__(self):
        if tk is None:
            print("tkinter not available. Please install tkinter for GUI support.")
            return

        self.root = tk.Tk()
        self.root.title("Snake AI Model Demo")
        self.root.geometry("600x700")

        # Models directory
        self.models_dir = Path(__file__).parent / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

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

        # Models section
        models_frame = ttk.LabelFrame(self.root, text="Available Models", padding=10)
        models_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Refresh button
        refresh_btn = tk.Button(models_frame, text="Refresh Models",
                               command=self.scan_models)
        refresh_btn.pack(anchor=tk.W, pady=5)

        # Single model dropdown (for single model demo)
        single_frame = tk.Frame(models_frame)
        single_frame.pack(fill=tk.X, pady=5)
        tk.Label(single_frame, text="Single Model:").pack(side=tk.LEFT, padx=5)
        self.single_model_var = tk.StringVar()
        self.single_model_combo = ttk.Combobox(single_frame, textvariable=self.single_model_var,
                                                state="readonly", width=50)
        self.single_model_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.single_model_combo.bind('<<ComboboxSelected>>', self.on_single_model_select)

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
        self.model_info_label = tk.Label(models_frame, text="Select a model to see details",
                                         wraplength=550, justify=tk.LEFT)
        self.model_info_label.pack(fill=tk.X, pady=5)

        # Settings section
        settings_frame = ttk.LabelFrame(self.root, text="Game Settings", padding=10)
        settings_frame.pack(fill=tk.X, padx=10, pady=5)

        # Grid size
        grid_frame = tk.Frame(settings_frame)
        grid_frame.pack(fill=tk.X, pady=2)
        tk.Label(grid_frame, text="Grid Size:").pack(side=tk.LEFT)
        self.grid_size_var = tk.StringVar(value="10")
        grid_spinbox = tk.Spinbox(grid_frame, from_=10, to=30, textvariable=self.grid_size_var, width=10)
        grid_spinbox.pack(side=tk.LEFT, padx=5)

        # Render delay/FPS
        fps_frame = tk.Frame(settings_frame)
        fps_frame.pack(fill=tk.X, pady=2)
        tk.Label(fps_frame, text="FPS (0=max speed):").pack(side=tk.LEFT)
        self.fps_var = tk.StringVar(value="10")
        fps_spinbox = tk.Spinbox(fps_frame, from_=0, to=60, textvariable=self.fps_var, width=10)
        fps_spinbox.pack(side=tk.LEFT, padx=5)

        # Show ray lines option (for ray models)
        ray_frame = tk.Frame(settings_frame)
        ray_frame.pack(fill=tk.X, pady=2)
        self.show_ray_lines_var = tk.BooleanVar(value=True)
        ray_checkbox = tk.Checkbutton(ray_frame, text="Show Ray Lines (for ray models)",
                                      variable=self.show_ray_lines_var)
        ray_checkbox.pack(side=tk.LEFT)

        # Buttons section
        buttons_frame = tk.Frame(self.root)
        buttons_frame.pack(fill=tk.X, padx=10, pady=10)

        self.start_single_btn = tk.Button(buttons_frame, text="Start Single Model Demo",
                                         command=self.start_single_demo,
                                         bg="#4CAF50", fg="white", font=("Arial", 10, "bold"),
                                         state=tk.DISABLED)
        self.start_single_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        self.start_pvp_btn = tk.Button(buttons_frame, text="Start PvP Demo",
                                       command=self.start_pvp_demo,
                                       bg="#2196F3", fg="white", font=("Arial", 10, "bold"),
                                       state=tk.DISABLED)
        self.start_pvp_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # Status/Log section
        log_frame = ttk.LabelFrame(self.root, text="Status", padding=5)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.log_text = scrolledtext.ScrolledText(log_frame, height=8, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log("Ready. Select models and click Start to begin.")

    def log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def scan_models(self):
        self.log("Scanning models directory...")
        self.single_model_combo['values'] = []
        self.model_list = []
        self.model_paths = []

        if not self.models_dir.exists():
            self.log(f"⚠ Models directory not found: {self.models_dir}")
            return

        models = list_models_in_directory(self.models_dir)

        if not models:
            self.log(f"⚠ No model files found in: {self.models_dir}")
            return

        model_items = list(models.items())
        display_names = []
        for filename, metadata in model_items:
            display_name = filename
            if metadata:
                if metadata.hidden_size2 is not None:
                    display_name += f" ({metadata.model_type.upper()}, {metadata.input_size}→{metadata.hidden_size}→{metadata.hidden_size2}→{metadata.output_size})"
                else:
                    display_name += f" ({metadata.model_type.upper()}, {metadata.input_size}→{metadata.hidden_size}→{metadata.output_size})"
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
                self.model_info_label.config(text=info)
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
                self.model_info_label.config(text=info)
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
            grid_size = int(self.grid_size_var.get())
            fps = int(self.fps_var.get())
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
                self.log(f"✓ Inferred: {format_model_info(metadata, filename)}")
            else:
                self.log(f"⚠ Could not infer architecture, using defaults")
                metadata = ModelMetadata("pvp", 17, 64, 3, "")

        self.log(f"Starting single model demo: {filename}")
        self.log("Game window will open. Close it when done.")
        print(f"DEBUG: Starting single model demo with {filename}, num_snakes=1")

        # Get show ray lines option
        show_ray_lines = self.show_ray_lines_var.get()

        # Run in separate thread
        thread = threading.Thread(target=demo_single_model,
                                 args=(model_path, metadata, grid_size, render_delay, True, show_ray_lines),
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
            grid_size = int(self.grid_size_var.get())
            fps = int(self.fps_var.get())
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