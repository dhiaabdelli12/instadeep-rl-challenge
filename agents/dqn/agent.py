"""
DQNAgent implementation module.
"""
from typing import Optional
from gym.wrappers.time_limit import TimeLimit
import numpy as np
import torch
from torch import optim
from agents.dqn.qnetwork import QNetwork


class DQNAgent:
    """DQNAgent class for implementing a Deep Q-Learning agent."""

    def __init__(
        self, env: TimeLimit, checkpoint_path: Optional[str] = None, **kwargs
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
        if checkpoint_path:
            self.qnetwork = torch.load(checkpoint_path)
            self.epsilon = self.epsilon_end
        else:
            self.qnetwork = QNetwork(self.s_dim, self.a_dim, self.device)
            self.epsilon = self.epsilon_start

        self.optimizer = optim.Adam(self.qnetwork.parameters(), lr=self.alpha)
        self.mem_cntr = 0
        self.iter_cntr = 0
        self.state_memory = np.zeros((self.mem_size, self.s_dim), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, self.s_dim), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)

    def _update_epsilon(self):
        """Updates the exploration-exploitation parameter epsilon."""
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
        if np.random.rand() > self.epsilon:
            state = torch.tensor(np.array([state]), device=self.qnetwork.device)
            actions = self.qnetwork.forward(state)
            action = torch.argmax(actions).item()
            return action
        return np.random.choice(np.arange(self.a_dim, dtype=np.int32))

    def learn(self):
        """Updates the QNetwork parameters based on sample batch from experience replay."""
        if self.mem_cntr < self.batch_size:
            return

        self.optimizer.zero_grad()
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = torch.tensor(
            self.state_memory[batch], device=self.qnetwork.device
        )
        new_state_batch = torch.tensor(
            self.new_state_memory[batch], device=self.qnetwork.device
        )
        action_batch = self.action_memory[batch]
        reward_batch = torch.tensor(
            self.reward_memory[batch], device=self.qnetwork.device
        )
        terminal_batch = torch.tensor(
            self.terminal_memory[batch], device=self.qnetwork.device
        )

        q_eval = self.qnetwork.forward(state_batch)[batch_index, action_batch]
        q_next = self.qnetwork.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]
        loss = self.qnetwork.loss_fn(q_target, q_eval)
        loss.backward()
        self.qnetwork.loss = loss
        self.optimizer.step()

        self.iter_cntr += 1
        self._update_epsilon()

    def store_transition(
        self,
        state: np.array,
        action: np.array,
        reward: float,
        next_state: np.array,
        done: np.bool_,
    ):
        """Stores a transition in the replay buffer.

        Parameters
        ----------
        state : np.array
            Current state of the agent.
        action : np.array
            Action taken by agent.
        reward : float
            Reward after taking action in a certain state.
        next_state : np.array
            State resulting from taking an action.
        done : np.bool_
            Whether or not the episode is finished.
        """
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = next_state
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def __str__(self) -> str:
        return f"""
        DQNAgent params:
        \t epsilon: {self.epsilon}
        \t QNetwork: {self.qnetwork}
        """
