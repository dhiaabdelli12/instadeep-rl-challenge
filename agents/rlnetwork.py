"""
Implementation of the base network class for the RL agent.
"""
import os
import torch
from torch import nn


class RLNetwork(nn.Module):
    """Base RL network class from which all networks inherit from."""

    def __init__(
        self,
        name: str,
        lr: float,
        input_dim: int,
        fc1_dim: int,
        fc2_dim: int,
        output_dim: int,
        device: str,
    ) -> None:
        """
        Initialize the RLNetwork.

        Parameters
        ----------
        name : str
            Name of the RLNetwork.
        lr : float
            Learning rate for the optimizer.
        input_dim : int
            Dimension of the input data.
        fc1_dim : int
            Dimension of the first fully connected layer.
        fc2_dim : int
            Dimension of the second fully connected layer.
        output_dim : int
            Dimension of the output data.
        device : str
            Device on which the network is allocated
        """
        super(RLNetwork, self).__init__()
        self.name = name
        self.input_dim = input_dim
        self.fc1_dim = fc1_dim
        self.fc2_dim = fc2_dim
        self.output_dim = output_dim
        self.lr = lr

        self.device = device
        self.loss = None

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the RLNetwork.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Raises
        ------
        NotImplementedError
            Method not implemented.
        """
        raise NotImplementedError("Method not implemented.")

    def save_checkpoint(self, chkpt_dir: str) -> None:
        """
        Save the network checkpoint.

        Parameters
        ----------
        chkpt_dir : str
            Directory to save the checkpoint.
        """
        path = os.path.join(chkpt_dir, f"{self.name}.pth")
        torch.save(self, path)
