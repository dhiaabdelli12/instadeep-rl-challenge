"""
QNetwork implementation module.
"""
import torch
from torch import nn
import torch.nn.functional as F


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
        self.fc1 = nn.Linear(s_dim, 512)
        self.V = nn.Linear(512, 1)
        self.A = nn.Linear(512, a_dim)

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
        flat1 = F.relu(self.fc1(x))
        V = self.V(flat1)
        A = self.A(flat1)

        return V, A



