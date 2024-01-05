"""
Implementation of Experience Replay
"""
import torch
import numpy as np

class ReplayBuffer:
    """Implementaion of the replay buffer for the experience replay."""

    def __init__(self, capacity: int, state_dim: int, device: str):
        """Replay Buffer class constructor.

        Parameters
        ----------
        capacity : int
            Maximum capacity of the replay buffer.
        state_dim : int
            Dimension of the state space.
        device : str
            Device on which replay buffer data is stored.
        """
        self.capacity = capacity
        self.device = device

        self.states = torch.zeros(
            (capacity, state_dim), dtype=torch.float32, device=device
        )
        self.next_states = torch.zeros(
            (capacity, state_dim), dtype=torch.float32, device=device
        )
        self.actions = torch.zeros(capacity, dtype=torch.int64, device=device)
        self.rewards = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.terminals = torch.zeros(capacity, dtype=torch.bool, device=device)

        self.index = 0
        self.size = 0

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: np.bool_,
    ):
        """Stores a transition in the replay buffer.

        Parameters
        ----------
        state : np.ndarray
            Current state of agent
        action : int
            Action taken
        reward : float
            Reward from taking action
        next_state : np.ndarray
            Next state after action is taken
        done : np.bool
            Whether the episode ended or not
        """
        index = self.index % self.capacity
        self.states[index] = torch.tensor(
            state, dtype=torch.float32, device=self.device
        )
        self.next_states[index] = torch.tensor(
            next_state, dtype=torch.float32, device=self.device
        )
        self.actions[index] = torch.tensor(
            action, dtype=torch.int32, device=self.device
        )
        self.rewards[index] = torch.tensor(
            reward, dtype=torch.float32, device=self.device
        )
        self.terminals[index] = torch.tensor(done, dtype=torch.bool, device=self.device)

        self.index += 1
        self.size = min(self.index, self.capacity)

    def sample_batch(self, batch_size: int):
        """Samples a batch from the replay buffer.

        Parameters
        ----------
        batch_size : int
            Batch size to sample from the buffer.

        Returns
        -------
        tuple
            sampled batches
        """
        indices = np.random.choice(
            min(self.index, self.capacity), batch_size, replace=False
        )
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.terminals[indices],
        )

    def clear(self):
        """Clears the replay buffer"""
        self.index = 0
        self.size = 0


class ContinuousReplayBuffer:
    """Implementaion of the replay buffer for the experience replay for continuous action space"""

    def __init__(self, capacity: int, state_dim: int, device: str):
        """Replay Buffer class constructor.

        Parameters
        ----------
        capacity : int
            Maximum capacity of the replay buffer.
        state_dim : int
            Dimension of the state space.
        device : str
            Device on which replay buffer data is stored.
        """
        self.capacity = capacity
        self.device = device

        self.states = torch.zeros(
            (capacity, state_dim), dtype=torch.float32, device=device
        )
        self.next_states = torch.zeros(
            (capacity, state_dim), dtype=torch.float32, device=device
        )
        self.actions = torch.zeros((capacity, 2), dtype=torch.float, device=device)
        self.rewards = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.terminals = torch.zeros(capacity, dtype=torch.bool, device=device)

        self.index = 0
        self.size = 0

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: np.bool_,
    ):
        """Stores a transition in the replay buffer.

        Parameters
        ----------
        state : np.ndarray
            Current state of agent
        action : int
            Action taken
        reward : float
            Reward from taking action
        next_state : np.ndarray
            Next state after action is taken
        done : np.bool
            Whether the episode ended or not
        """
        index = self.index % self.capacity

        self.states[index] = torch.tensor(
            state, dtype=torch.float32, device=self.device
        )
        self.next_states[index] = torch.tensor(
            next_state, dtype=torch.float32, device=self.device
        )
        self.actions[index] = torch.tensor(
            action, dtype=torch.float, device=self.device
        )
        self.rewards[index] = torch.tensor(
            reward, dtype=torch.float32, device=self.device
        )
        self.terminals[index] = torch.tensor(done, dtype=torch.bool, device=self.device)

        self.index += 1
        self.size = min(self.index, self.capacity)

    def sample_batch(self, batch_size: int):
        """Samples a batch from the replay buffer.

        Parameters
        ----------
        batch_size : int
            Batch size to sample from the buffer.

        Returns
        -------
        tuple
            sampled batches
        """
        indices = np.random.choice(
            min(self.index, self.capacity), batch_size, replace=False
        )
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.terminals[indices],
        )

    def clear(self):
        """Clears the replay buffer"""
        self.index = 0
        self.size = 0