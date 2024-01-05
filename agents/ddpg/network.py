"""
Implementation of the Actor and Critic networks
"""
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from agents.rlnetwork import RLNetwork


class Critic(RLNetwork):
    """
    Implementation of the Critic network class.
    """

    def __init__(
        self,
        name: str,
        lr: float,
        input_dim: int,
        fc1_dim: int,
        fc2_dim: int,
        output_dim: int,
        device: str,
        n_action: int,
    ) -> None:
        """
        Initializes the Critic network

        Parameters
        ----------
        name : str
            Name of the network.
        lr : float
            Learning rate for the optimizer.
        input_dim : int
            Dimension of the input data.
        fc1_dim : int
            Dimension of first fully connected layer.
        fc2_dim : int
            Dimension of second fully connected layer.
        output_dim : int
            Dimension of the output layer.
        device : str
            Device on which the network is allocated.
        n_action : int
            Dimension of action space.
        """
        super().__init__(name, lr, input_dim, fc1_dim, fc2_dim, output_dim, device)
        self.fc1 = nn.Linear(self.input_dim, self.fc1_dim)
        self.bn1 = nn.LayerNorm(self.fc1_dim)
        self.fc2 = nn.Linear(self.fc1_dim, self.fc2_dim)
        self.bn2 = nn.LayerNorm(self.fc2_dim)
        self.action_value = nn.Linear(n_action, fc2_dim)
        self.q = nn.Linear(self.fc2_dim, 1)
        self._init_weights()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(device)

    def _init_weights(self):
        """
        Initiliazes weights and biases for layers.
        """
        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        f3 = 0.003
        nn.init.uniform_(self.q.weight.data, -f3, f3)
        nn.init.uniform_(self.q.bias.data, -f3, f3)

    def forward(self, state: torch.Tensor, action: int) -> float:
        """
        Forward pass of the network.

        Parameters
        ----------
        state : torch.Tensor
            Current state.
        action : int
            Action taken.

        Returns
        -------
        float
            Q value for state action pair.
        """
        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)

        action_value = F.relu(self.action_value(action))
        state_action_value = F.relu(torch.add(state_value, action_value))
        state_action_value = self.q(state_action_value)

        return state_action_value


class Actor(RLNetwork):
    """
    Implementation of the Critic network class.
    """

    def __init__(
        self,
        name: str,
        lr: float,
        input_dim: int,
        fc1_dim: int,
        fc2_dim: int,
        output_dim: int,
        device: str,
        n_action: int,
    ) -> None:
        """
        Initializes the Actor network.

        Parameters
        ----------
        name : str
            Name of the network.
        lr : float
            Learning rate for the optimizer.
        input_dim : int
            Dimension of the input data.
        fc1_dim : int
            Dimension of first fully connected layer.
        fc2_dim : int
            Dimension of second fully connected layer.
        output_dim : int
            Dimension of the output layer.
        device : str
            Device on which the network is allocated.
        n_action : int
            Dimension of action space.
        """
        super().__init__(name, lr, input_dim, fc1_dim, fc2_dim, output_dim, device)
        self.fc1 = nn.Linear(self.input_dim, self.fc1_dim)
        self.bn1 = nn.LayerNorm(self.fc1_dim)
        self.fc2 = nn.Linear(self.fc1_dim, self.fc2_dim)
        self.bn2 = nn.LayerNorm(self.fc2_dim)
        self.mu = nn.Linear(self.fc2_dim, n_action)
        self._init_weights()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(device=device)

    def _init_weights(self):
        """
        Initiliazes weights and biases for layers.
        """
        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        f3 = 0.003
        nn.init.uniform_(self.mu.weight.data, -f3, f3)
        nn.init.uniform_(self.mu.bias.data, -f3, f3)

    def forward(self, x):
        """
        Forward pass of the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.tanh(self.mu(x))
        return x
