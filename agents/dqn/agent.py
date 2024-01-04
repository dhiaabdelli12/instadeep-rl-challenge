"""
DQNAgent implementation module.
"""
import os
import logging
from typing import Optional
from gym.wrappers.time_limit import TimeLimit
import numpy as np
import torch
from torch import optim
from agents.dqn.qnetwork import QNetwork
from datetime import datetime
from pathlib import Path


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


class DQNAgent:
    """DQNAgent class for implementing a Deep Q-Learning agent."""

    def __init__(
        self,
        env: TimeLimit,
        verbose=True,
        checkpoint_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        """DQNAgent class constructor.

        Parameters
        ----------
        checkpoint_path : None | str, optional
            path of the saved qnetwork checkpoint, by default None
        """
        self.__dict__.update(kwargs)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.s_dim = env.observation_space.shape[0]
        self.a_dim = env.action_space.n
        self.replay_buffer = ReplayBuffer(self.mem_size, self.s_dim, self.device)
        self.learn_step_cnt = 0
        self.checkpoint_path = checkpoint_path
        self.verbose = verbose

        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)
        else:
            self.qnetwork = QNetwork(self.s_dim, self.a_dim, self.device)
            self.epsilon = self.epsilon_start
            self.optimizer = optim.Adam(self.qnetwork.parameters(), lr=self.alpha)

        if verbose:
            logger.info("Agent initialized.")

    def _load_checkpoint(self, path):
        self.checkpoint_name = path.split("/")[-2]
        self.qnetwork = torch.load(f"{path}/qnetwork.pth", map_location=self.device)
        self.epsilon = self.epsilon_end
        if self.verbose:
            logger.info("Loaded QNetwork from checkpoint: %s", path)
            logger.info(self.__str__())

    def _update_epsilon(self):
        """Updates the exploration-exploitation parameter epsilon."""
        with torch.no_grad():
            self.epsilon = (
                self.epsilon - self.epsilon_decay
                if self.epsilon > self.epsilon_end
                else self.epsilon_end
            )

    def act(self, state: np.ndarray) -> np.ndarray:
        """Chooses action based on epsilon-greedy policy.

        Parameters
        ----------
        state : np.ndarray
            The current state of the agent.

        Returns
        -------
        np.ndarray
            The chosen action based on the policy.
        """
        if torch.rand(1) > self.epsilon:
            state = torch.tensor(state, device=self.qnetwork.device)
            actions = self.qnetwork.forward(state)
            action = torch.argmax(actions).item()
            return action
        return np.random.choice(np.arange(self.a_dim, dtype=np.int32))

    def learn(self):
        """Updates the QNetwork parameters based on sample batch from experience replay."""
        if self.replay_buffer.index < self.batch_size:
            return

        self.optimizer.zero_grad()

        (
            state_batch,
            action_batch,
            reward_batch,
            new_state_batch,
            terminal_batch,
        ) = self.replay_buffer.sample_batch(self.batch_size)

        q_eval = self.qnetwork.forward(state_batch)[
            np.arange(self.batch_size), action_batch
        ]
        q_next = self.qnetwork.forward(new_state_batch).max(dim=1)[0]
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * q_next
        loss = self.qnetwork.loss_fn(q_target, q_eval)
        loss.backward()
        self.qnetwork.loss = loss
        self.optimizer.step()
        self._update_epsilon()

    def save_checkpoint(self, iteration: int):
        """Saves QNetwork checkpoint at a specific iteration.

        Parameters
        ----------
        network : Agent
            Agent in training.
        iteration : int
            Iteration number at each network weights will be saved.
        """
        timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M")
        chkpt_name = f"{iteration}_eps-{timestamp}"

        root_directory = Path(__file__).parent.parent.parent
        chkpt_dir_path = os.path.join(root_directory, "checkpoints", "dqn", chkpt_name)
        print(chkpt_dir_path)
        qnetwork_path = os.path.join(chkpt_dir_path, "qnetwork.pth")
        os.makedirs(chkpt_dir_path, exist_ok=True)
        torch.save(self.qnetwork, qnetwork_path)

        return chkpt_dir_path

    def __str__(self) -> str:
        return f"""
        DQNAgent params:
        \t epsilon: {self.epsilon}
        \t QNetwork: {self.qnetwork}
        """
