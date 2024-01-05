""""
Implemented the QNetwork for deep Q-learning.
"""
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from agents.rlnetwork import RLNetwork


class QNetwork(RLNetwork):
    """
    Implementation of the QNetwork class.
    """

    def __init__(
        self, name: str, lr: float, input_dim: int, output_dim: int, device: str
    ) -> None:
        """
        Initialize the QNetwork.

        Parameters
        ----------
        name : str
            Name of the QNetwork.
        lr : float
            Learning rate for the optimizer.
        input_dim : int
            Dimension of the input data.
        output_dim : int
            Dimension of the output data.
        device : torch.device
            Device on which the network is allocated.
        """
        super().__init__(
            name=name,
            lr=lr,
            input_dim=input_dim,
            fc1_dim=256,
            fc2_dim=256,
            output_dim=output_dim,
            device=device,
        )
        self.fc1 = nn.Linear(self.input_dim, self.fc1_dim)
        self.fc2 = nn.Linear(self.fc1_dim, self.fc2_dim)
        self.fc3 = nn.Linear(self.fc2_dim, self.output_dim)
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the QNetwork.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        x = x.to(self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
