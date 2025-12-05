import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self, input_size=17, hidden_size=64, output_size=3, device=None, hidden_size2=None):
        super(NeuralNetwork, self).__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_size2 = hidden_size2  # For genetic models with two hidden layers

        if hidden_size2 is not None:
            # Genetic model architecture: (input_size, hidden_size, hidden_size2, output_size)
            # 3 layers: fc1, fc2, fc3
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size2)
            self.fc3 = nn.Linear(hidden_size2, output_size)
            self.fc4 = None
            self.dropout = None  # Genetic models don't use dropout
        else:
            # Standard architecture: (input_size, hidden_size, hidden_size, hidden_size//2, output_size)
            # 4 layers: fc1, fc2, fc3, fc4
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
            self.fc4 = nn.Linear(hidden_size // 2, output_size)
            # Add dropout for regularization during evolution
            self.dropout = nn.Dropout(0.1)

        # Move to device
        self.to(self.device)

    def forward(self, x, return_activations=False):
        if self.hidden_size2 is not None:
            # Genetic model: 3 layers with ReLU on hidden, sigmoid on output
            fc1_out = torch.relu(self.fc1(x))
            fc1_activations = fc1_out.clone()

            fc2_out = torch.relu(self.fc2(fc1_out))
            fc2_activations = fc2_out.clone()

            output = torch.sigmoid(self.fc3(fc2_out))
            output_activations = output.clone()

            if return_activations:
                return output, {
                    'fc1': fc1_activations,
                    'fc2': fc2_activations,
                    'output': output_activations
                }
            return output
        else:
            # Standard model: 4 layers with ReLU on hidden, linear on output
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
        self.eval()  # Set to evaluation mode to disable dropout
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


class ConvQN(nn.Module):
    # Convolutional Q-Network for pixel-based Snake models
    def __init__(self, in_channels=5, extra_dim=8, num_actions=3, feat_dim=64, hidden=128, device=None):
        super(ConvQN, self).__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.convLayers = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, feat_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)  # -> [B, feat_dim, 1, 1]
        self.head = nn.Sequential(
            nn.Linear(feat_dim + extra_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_actions)
        )

        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)

        # Move to device
        self.to(self.device)

    def forward(self, planes, direction, return_activations=False):
        h = self.convLayers(planes)
        h = self.pool(h).flatten(1)

        # Store conv activations for visualization
        conv_activations = h.clone()

        h = torch.cat([h, direction], dim=-1)
        head_input = h.clone()

        # Pass through head
        head_hidden = self.head[0](h)
        head_hidden_activations = head_hidden.clone()
        head_hidden = torch.relu(head_hidden)
        q = self.head[2](head_hidden)
        output_activations = q.clone()

        if return_activations:
            return q, {
                'conv': conv_activations,
                'head_input': head_input,
                'head_hidden': head_hidden_activations,
                'output': output_activations
            }
        return q

    def act(self, state, return_activations=False):
        # Act on state. State should be a tuple (planes, direction) for pixel models
        self.eval()  # Set to evaluation mode to disable dropout
        with torch.no_grad():
            planes, direction = state
            # Ensure tensors are on correct device and have batch dimension
            if isinstance(planes, torch.Tensor):
                planes = planes.unsqueeze(0).to(self.device)
            else:
                planes = torch.FloatTensor(planes).unsqueeze(0).to(self.device)

            if isinstance(direction, torch.Tensor):
                direction = direction.unsqueeze(0).to(self.device)
            else:
                direction = torch.FloatTensor(direction).unsqueeze(0).to(self.device)

            if return_activations:
                output, activations = self.forward(planes, direction, return_activations=True)
                # Convert activations to CPU and squeeze batch dimension
                activations = {k: v.cpu().squeeze(0) for k, v in activations.items()}
                return output.argmax().item(), activations
            else:
                output = self.forward(planes, direction)
                return output.argmax().item()

