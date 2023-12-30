"""
QNetwork implementation module.
"""
import torch
from torch import nn


class QNetwork(nn.Module):
    """QNetwork class for defining a simple neural network for Q-learning."""

    def __init__(self, s_dim: int, a_dim: int, device: str) -> None:
        """QNetwork calss constructor.

        Parameters
        ----------
        s_dim : int
            Observation space dimension.
        a_dim : int
            Action space dimension.
        device : str
            device where model is stored and trained (cpu/cuda).
        """
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(s_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, a_dim)
        self.loss_fn = nn.MSELoss()
        self.loss = None
        self.device = device
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the neural network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor representing the state of the agent

        Returns
        -------
        torch.Tensor
            Porbability distribution over agent actions.
        """
        x = x.to(self.device)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
