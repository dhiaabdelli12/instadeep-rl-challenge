"""
Implementation of the Deep Q-learning agent.
"""
import torch
import gym
import numpy as np
from agents.rlagent import RLAgent
from agents.dqn.network import QNetwork


class DQNAgent(RLAgent):
    """
    Deep Q-learning implementation class.
    """

    def __init__(
        self, name: str, env: gym.Env, chkpt_path: None | str = None, **kwargs
    ) -> None:
        """
        Initialize the DQNAgent.

        Parameters
        ----------
        name : str
            Name of the DQNAgent.
        env : gym.Env
            Environment for the DQNAgent.
        chkpt_path : str or None, optional
            Path to load checkpoint, by default None.
        **kwargs
            Additional parameters to be set as attributes of the DQNAgent.
        """
        super().__init__(name, env, chkpt_path, **kwargs)
        self.a_dim = self.env.action_space.n
        if not chkpt_path:
            self.networks["qnetwork"] = QNetwork(
                name="qnetwork",
                lr=self.alpha,
                input_dim=self.s_dim,
                output_dim=self.a_dim,
                device=self.device,
            )
            self.epsilon = self.epsilon_start
        self.logger = self._init_logger("DQNAgent")

    def _update_epsilon(self):
        """
        Updates the exploration-exploitation parameter epsilon.
        """
        with torch.no_grad():
            self.epsilon = (
                self.epsilon - self.epsilon_decay
                if self.epsilon > self.epsilon_end
                else self.epsilon_end
            )

    def act(self, state: np.ndarray) -> int:
        """
        Select an action using the epsilon-greedy strategy.

        Parameters
        ----------
        state : np.ndarray
            Current state.

        Returns
        -------
        int
            Selected action.
        """
        if torch.rand(1) > self.epsilon:
            state = torch.tensor(state, device=self.device)
            actions = self.networks["qnetwork"](state)
            action = torch.argmax(actions).item()
            return action
        return np.random.choice(np.arange(self.a_dim, dtype=np.int32))

    def learn(self):
        """
        Implements the learning algorithm for the DQNAgent.
        """
        if self.replay_buffer.index < self.batch_size:
            return
        (
            state_batch,
            action_batch,
            reward_batch,
            new_state_batch,
            terminal_batch,
        ) = self.replay_buffer.sample_batch(self.batch_size)
        self.networks["qnetwork"].optimizer.zero_grad()
        q_eval = self.networks["qnetwork"](state_batch)[
            np.arange(self.batch_size), action_batch
        ]
        q_next = self.networks["qnetwork"](new_state_batch).max(dim=1)[0]
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * q_next
        loss = self.networks["qnetwork"].loss_fn(q_target, q_eval)
        self.networks["qnetwork"].loss = torch.Tensor.cpu(loss.detach()).numpy()
        loss.backward()
        self.networks["qnetwork"].optimizer.step()
        self._update_epsilon()

    def __str__(self):
        """
        String representation of the class.
        """
        return super().__str__()
