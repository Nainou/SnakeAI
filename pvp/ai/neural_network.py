"""
Neural Network Architecture for PvP Snake AI

Enhanced neural network designed for competitive snake gameplay.
"""

import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self, input_size=17, hidden_size=64, output_size=3, device=None):
        """
        Enhanced neural network for PvP genetic algorithm

        Args:
            input_size: Size of input state vector (17 for snake state)
            hidden_size: Size of hidden layers
            output_size: Number of actions (3: straight, right, left)
            device: Device to run on (cuda/cpu)
        """
        super(NeuralNetwork, self).__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Larger, deeper network for better learning capacity
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, output_size)

        # Add dropout for regularization during evolution
        self.dropout = nn.Dropout(0.1)

        # Move to device
        self.to(self.device)

    def forward(self, x):
        """Forward pass through the network"""
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def act(self, state):
        """Get action from the neural network"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            output = self.forward(state_tensor)
            return output.argmax().item()
