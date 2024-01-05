"""
Implementation of the Double Deep Q-learning agent.
"""
import torch
import gym
import numpy as np
from agents.ddqn.network import QNetwork
from agents.rlagent import RLAgent


class DDQNAgent(RLAgent):
    """
    Implementation of the Double Deep Q-learning class.
    """

    def __init__(
        self, name: str, env: gym.Env, chkpt_path: str | None = None, **kwargs
    ) -> None:
        """
        Initialize the DDQNAgent.

        Parameters
        ----------
        name : str
            Name of the DDQNAgent.
        env : gym.Env
            Environment for the DDQNAgent.
        chkpt_path : str or None, optional
            Path to load checkpoint, by default None.
        **kwargs
            Additional parameters to be set as attributes of the DDQNAgent.
        """
        super().__init__(name, env, chkpt_path, **kwargs)
        self.a_dim = self.env.action_space.n
        if not chkpt_path:
            self.networks = {
                "q_eval": QNetwork(
                    name="q_eval",
                    lr=self.alpha,
                    input_dim=self.s_dim,
                    output_dim=self.a_dim,
                    device=self.device,
                ),
                "q_next": QNetwork(
                    name="q_next",
                    lr=self.alpha,
                    input_dim=self.s_dim,
                    output_dim=self.a_dim,
                    device=self.device,
                ),
            }
            self.epsilon = self.epsilon_start
        self.logger = self._init_logger("DDQNAgent")
        self.learn_step_counter = 0

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

    def act(self, state: np.ndarray) -> np.ndarray:
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
            state = torch.tensor(state, dtype=torch.float, device=self.device)
            _, advantage = self.networks["q_eval"](state)
            action = torch.argmax(advantage).item()
            return action
        return np.random.choice(np.arange(self.a_dim, dtype=np.int32))

    def _replace_target_network(self):
        """
        Replaces the target Q-network with the evaluation Q-network periodically.
        """
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.networks["q_next"].load_state_dict(
                self.networks["q_eval"].state_dict()
            )

    def learn(self):
        """
        Implements the learning algorithm for the DQNAgent.
        """
        if self.replay_buffer.index < self.batch_size:
            return

        self.networks["q_eval"].optimizer.zero_grad()
        self.networks["q_next"].optimizer.zero_grad()

        self._replace_target_network()

        (
            state_batch,
            action_batch,
            reward_batch,
            new_state_batch,
            terminal_batch,
        ) = self.replay_buffer.sample_batch(self.batch_size)
        indices = np.arange(self.batch_size)

        v, a = self.networks["q_eval"].forward(state_batch)
        v_next, a_next = self.networks["q_next"].forward(new_state_batch)
        v_eval, a_eval = self.networks["q_eval"].forward(new_state_batch)

        q_pred = torch.add(v, (a - a.mean(dim=1, keepdim=True)))[indices, action_batch]
        q_next = torch.add(v_next, (a_next - a_next.mean(dim=1, keepdim=True)))
        q_eval = torch.add(v_eval, (a_eval - a_eval.mean(dim=1, keepdim=True)))

        max_actions = torch.argmax(q_eval, dim=1)
        q_next[terminal_batch] = 0.0
        q_target = reward_batch + self.gamma * q_next[indices, max_actions]

        loss = self.networks["q_eval"].loss_fn(q_target, q_pred).to(self.device)
        loss.backward()
        self.networks["q_eval"].loss = torch.Tensor.cpu(loss.detach()).numpy()
        self.networks["q_eval"].optimizer.step()
        self.learn_step_counter += 1
        self._update_epsilon()

    def __str__(self) -> str:
        """
        String representation of the class.
        """
        return f"""
        DDQNAgent params:
        \t epsilon: {self.epsilon}
        \t QEval Network: {self.q_eval}
        \t QNext Network: {self.q_next}
        """
