"""
PvP Snake Game Module

This module contains the core game logic for the PvP Snake game where 2-4 snakes compete.
"""

from .snake_game_pvp import SnakeGamePvP, Direction
from .snake import Snake

__all__ = ['SnakeGamePvP', 'Snake', 'Direction']
