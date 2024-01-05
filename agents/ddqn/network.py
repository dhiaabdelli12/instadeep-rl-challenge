""""
Implemented the networks for double deep Q-learning.
"""
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from agents.rlnetwork import RLNetwork


class QNetwork(RLNetwork):
    """
    Implementation of the q_eval/q_next network class.
    """

    def __init__(self, name, lr, input_dim, output_dim, device) -> None:
        """
        Initialize the Network.

        Parameters
        ----------
        name : str
            Name of the network.
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
            fc1_dim=512,
            fc2_dim=512,
            output_dim=output_dim,
            device=device,
        )
        self.fc1 = nn.Linear(input_dim, self.fc1_dim)
        self.V = nn.Linear(self.fc1_dim, 1)
        self.A = nn.Linear(self.fc2_dim, output_dim)
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """_summary_

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Value action pairs.
        """
        flat1 = F.relu(self.fc1(x))
        V = self.V(flat1)
        A = self.A(flat1)
        return V, A
