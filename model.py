"""
Neural network architectures for the Super Mario Agent.
Includes Spatial Transformer Network, multi-branch DQN, and Rainbow enhancements.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SpatialTransformer(nn.Module):
    """Spatial Transformer Network to learn attention on relevant screen regions."""
    
    def __init__(self, in_channels):
        super().__init__()
        # Localization network
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        
        # Regressor for affine transformation parameters (6 parameters)
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 18 * 18, 32),
            nn.ReLU(True),
            nn.Linear(32, 6)
        )
        
        # Initialize weights/bias for identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
    
    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 18 * 18)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)
        return x


class NoisyLinear(nn.Module):
    """Noisy linear layer for parameterized exploration."""
    
    def __init__(self, in_features, out_features, std_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))
    
    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())
    
    def forward(self, input):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(input, weight, bias)


class RainbowDQN(nn.Module):
    """
    Rainbow DQN with:
    - Multi-branch architecture (current frame, screenshot history, action history)
    - Spatial Transformer Network
    - Dueling architecture
    - Distributional RL (C51)
    - Noisy networks for exploration
    """
    
    def __init__(self, in_channels=4, num_actions=7, num_atoms=51, 
                 v_min=-10, v_max=10, use_noisy=True, use_stn=True):
        super().__init__()
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.use_noisy = use_noisy
        self.use_stn = use_stn
        
        # Register atoms for distributional RL
        self.register_buffer('atoms', torch.linspace(v_min, v_max, num_atoms))
        
        # Spatial Transformer Network
        if use_stn:
            self.stn = SpatialTransformer(in_channels)
            stn_out_channels = in_channels
        else:
            self.stn = nn.Identity()
            stn_out_channels = in_channels
        
        # Current frame branch (CNN)
        self.current_branch = nn.Sequential(
            nn.Conv2d(stn_out_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Screenshot history branch (shared or separate CNN)
        self.history_branch = nn.Sequential(
            nn.Conv2d(stn_out_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Calculate flattened size (84x84 -> 20x20 -> 9x9 -> 7x7 after convs)
        conv_output_size = 64 * 7 * 7
        
        # Action history branch (LSTM)
        self.action_embedding = nn.Embedding(num_actions, 32)
        self.action_lstm = nn.LSTM(32, 64, batch_first=True)
        action_history_size = 64
        
        # Merge all branches
        merged_size = conv_output_size * 2 + action_history_size  # current + history + action
        
        # Dueling architecture: Value and Advantage streams
        if use_noisy:
            Linear = NoisyLinear
        else:
            Linear = nn.Linear
        
        self.value_stream = nn.Sequential(
            Linear(merged_size, 512),
            nn.ReLU(),
            Linear(512, num_atoms)
        )
        
        self.advantage_stream = nn.Sequential(
            Linear(merged_size, 512),
            nn.ReLU(),
            Linear(512, num_actions * num_atoms)
        )
    
    def forward(self, state, action_history=None):
        """
        Forward pass through the network.
        
        Args:
            state: Current state tensor [batch, channels, height, width]
            action_history: Tensor of previous actions [batch, history_length]
        
        Returns:
            q_values: Q-values for each action [batch, num_actions]
            probs: Probability distribution over atoms [batch, num_actions, num_atoms]
        """
        batch_size = state.size(0)
        
        # Apply STN if enabled
        if self.use_stn:
            state = self.stn(state)
        
        # Current frame branch
        current_features = self.current_branch(state)
        current_features = current_features.view(batch_size, -1)
        
        # Screenshot history branch (using same state for now, can be extended)
        history_features = self.history_branch(state)
        history_features = history_features.view(batch_size, -1)
        
        # Action history branch
        if action_history is not None and action_history.size(1) > 0:
            action_embeds = self.action_embedding(action_history)
            lstm_out, _ = self.action_lstm(action_embeds)
            action_features = lstm_out[:, -1, :]  # Take last output
        else:
            # Default: zero vector if no history
            action_features = torch.zeros(batch_size, 64, device=state.device)
        
        # Merge all branches
        merged = torch.cat([current_features, history_features, action_features], dim=1)
        
        # Dueling architecture
        value = self.value_stream(merged).view(batch_size, 1, self.num_atoms)
        advantage = self.advantage_stream(merged).view(batch_size, self.num_actions, self.num_atoms)
        
        # Combine value and advantage
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        # Convert to probability distribution
        probs = F.softmax(q_atoms, dim=2)
        
        # Compute Q-values as expectation over atoms
        q_values = torch.sum(probs * self.atoms.unsqueeze(0).unsqueeze(0), dim=2)
        
        return q_values, probs
    
    def reset_noise(self):
        """Reset noise in all noisy linear layers."""
        if self.use_noisy:
            for module in self.modules():
                if isinstance(module, NoisyLinear):
                    module.reset_noise()


class SimpleDQN(nn.Module):
    """Simpler DQN variant without all Rainbow features (for comparison)."""
    
    def __init__(self, in_channels=4, num_actions=7):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

