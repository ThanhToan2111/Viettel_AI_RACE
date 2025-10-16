import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Actor(nn.Module):
    """
    Improved Actor network for PPO agent
    Outputs continuous actions (power ratios) for each cell
    Uses Gaussian policy with learned standard deviation

    Improvements:
    - Wider network (512 hidden dim)
    - Layer Normalization for stable training
    - Residual connections
    - ELU activation (better than ReLU)
    - Dropout regularization
    """

    def __init__(self, state_dim, action_dim, hidden_dim=512, activation='elu', dropout=0.1):
        """
        Initialize Actor network

        Args:
            state_dim (int): Dimension of input state
            action_dim (int): Dimension of output action
            hidden_dim (int): Hidden layer dimension (increased to 512)
            activation (str): Activation function type (default 'elu')
            dropout (float): Dropout probability
        """
        super(Actor, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
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

        # Action mean head
        self.action_mean = nn.Linear(hidden_dim // 2, action_dim)

        # Action standard deviation head (learnable)
        self.action_logstd = nn.Linear(hidden_dim // 2, action_dim)

        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize network weights using orthogonal initialization"""
        for layer in [self.fc1, self.fc2, self.fc3, self.fc4]:
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.zeros_(layer.bias)

        # Initialize action mean with smaller weights for stability
        nn.init.orthogonal_(self.action_mean.weight, gain=0.01)
        nn.init.zeros_(self.action_mean.bias)

        # Initialize log std to produce reasonable initial exploration
        nn.init.constant_(self.action_logstd.weight, 0.0)
        nn.init.constant_(self.action_logstd.bias, -0.5)  # Initial std ≈ 0.6

    def forward(self, state):
        """
        Forward pass through improved actor network

        Args:
            state (torch.Tensor): Input state tensor

        Returns:
            tuple: (action_mean, action_logstd)
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

        # Compute action mean (bounded to [0, 1] using sigmoid)
        action_mean = torch.sigmoid(self.action_mean(x))

        # Compute action log standard deviation (bounded for stability)
        action_logstd = torch.clamp(self.action_logstd(x), min=-20, max=2)

        return action_mean, action_logstd