"""
AI Module for PvP Snake Game

This module contains the neural network architecture and genetic algorithm
for training multiple snakes to compete against each other.
"""

from .neural_network import NeuralNetwork
from .genetic_algorithm import GeneticAlgorithm, Individual

__all__ = ['NeuralNetwork', 'GeneticAlgorithm', 'Individual']
