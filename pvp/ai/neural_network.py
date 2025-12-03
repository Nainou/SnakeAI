import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self, input_size=17, hidden_size=64, output_size=3, device=None):
        # Neural network for PvP genetic algorithm
        # Args:
        #   input_size: Size of input state vector (17 for snake state)
        #   hidden_size: Size of hidden layers
        #   output_size: Number of actions (3: straight, right, left)
        #   device: Device to run on (cuda/cpu)
        super(NeuralNetwork, self).__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, output_size)

        # Add dropout for regularization during evolution
        self.dropout = nn.Dropout(0.1)

        # Move to device
        self.to(self.device)

    def forward(self, x, return_activations=False):
        # Forward pass through the network
        fc1_out = torch.relu(self.fc1(x))
        fc1_activations = fc1_out.clone()

        x = self.dropout(fc1_out)
        fc2_out = torch.relu(self.fc2(x))
        fc2_activations = fc2_out.clone()

        x = self.dropout(fc2_out)
        fc3_out = torch.relu(self.fc3(x))
        fc3_activations = fc3_out.clone()

        output = self.fc4(fc3_out)
        output_activations = output.clone()

        if return_activations:
            return output, {
                'fc1': fc1_activations,
                'fc2': fc2_activations,
                'fc3': fc3_activations,
                'output': output_activations
            }
        return output

    def act(self, state, return_activations=False):
        # Get action from the neural network
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            if return_activations:
                output, activations = self.forward(state_tensor, return_activations=True)
                # Convert activations to CPU and squeeze batch dimension
                activations = {k: v.cpu().squeeze(0) for k, v in activations.items()}
                return output.argmax().item(), activations
            else:
                output = self.forward(state_tensor)
                return output.argmax().item()
