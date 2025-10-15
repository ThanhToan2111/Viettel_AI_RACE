import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Critic(nn.Module):
    """
    Improved Critic network for PPO agent
    Estimates state values for policy gradient methods

    Improvements:
    - Wider network (512 hidden dim)
    - Layer Normalization for stable training
    - Residual connections
    - ELU activation (better than ReLU)
    - Dropout regularization
    """

    def __init__(self, state_dim, hidden_dim=512, activation='elu', dropout=0.1):
        """
        Initialize Critic network

        Args:
            state_dim (int): Dimension of input state
            hidden_dim (int): Hidden layer dimension (increased to 512)
            activation (str): Activation function type (default 'elu')
            dropout (float): Dropout probability
        """
        super(Critic, self).__init__()

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.dropout_prob = dropout

        # Choose activation function
        if activation == 'relu':
            self.activation_fn = F.relu
        elif activation == 'tanh':
            self.activation_fn = torch.tanh
        elif activation == 'elu':
            self.activation_fn = F.elu
        elif activation == 'gelu':
            self.activation_fn = F.gelu
        else:
            self.activation_fn = F.elu  # Default to ELU

        # Network layers with LayerNorm
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)

        self.fc4 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.ln4 = nn.LayerNorm(hidden_dim // 2)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Value head
        self.value_head = nn.Linear(hidden_dim // 2, 1)

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        """Initialize network weights using orthogonal initialization"""
        for layer in [self.fc1, self.fc2, self.fc3, self.fc4]:
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.zeros_(layer.bias)

        # Initialize value head with smaller weights
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, state):
        """
        Forward pass through improved critic network

        Args:
            state (torch.Tensor): Input state tensor

        Returns:
            torch.Tensor: Estimated state value
        """
        # Layer 1: Input processing
        x = self.fc1(state)
        x = self.ln1(x)
        x = self.activation_fn(x)
        x = self.dropout(x)

        # Layer 2: Deep processing with residual
        identity = x
        x = self.fc2(x)
        x = self.ln2(x)
        x = self.activation_fn(x)
        x = x + identity  # Residual connection
        x = self.dropout(x)

        # Layer 3: Deep processing with residual
        identity = x
        x = self.fc3(x)
        x = self.ln3(x)
        x = self.activation_fn(x)
        x = x + identity  # Residual connection
        x = self.dropout(x)

        # Layer 4: Output preparation
        x = self.fc4(x)
        x = self.ln4(x)
        x = self.activation_fn(x)

        # Value estimation
        value = self.value_head(x)

        return value