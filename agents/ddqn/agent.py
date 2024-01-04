import logging
import os
from datetime import datetime
from pathlib import Path
from gym.wrappers.time_limit import TimeLimit
import numpy as np
import torch
from torch import optim
from agents.ddqn.networks import QNetwork
from agents.dqn.agent import DQNAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DDQNAgent(DQNAgent):
    def __init__(
        self, env: TimeLimit, verbose=True, checkpoint_path: str | None = None, **kwargs
    ) -> None:
        super().__init__(env, verbose, checkpoint_path, **kwargs)
        self.learn_step_counter = 0
        print(checkpoint_path)
        if self.checkpoint_path:
           self._load_checkpoint(self.checkpoint_path) 
        else:
            self.q_eval = QNetwork(self.s_dim, self.a_dim, self.device)
            self.q_next = QNetwork(self.s_dim, self.a_dim, self.device)
            self.epsilon = self.epsilon_start

        self.q_eval_optimizer = optim.Adam(self.q_eval.parameters(), lr=self.alpha)
        self.q_next_optimizer = optim.Adam(self.q_next.parameters(), lr=self.alpha)
    
    def _load_checkpoint(self, path):
        q_eval_path = os.path.join(path, "q_eval.pth")
        q_next_path = os.path.join(path, "q_next.pth")
        
        
        self.q_eval = torch.load(q_eval_path, map_location=self.device)
        self.q_next = torch.load(q_next_path, map_location=self.device)

        self.q_eval.eval()
        self.q_next.eval()
        self.epsilon = self.epsilon_end
        if self.verbose:
            logger.info(
                "Loaded QEval and QNext from checkpoint %s", path
            )
            logger.info(self.__str__())

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
            state = torch.tensor(state, dtype=torch.float, device=self.q_eval.device)
            output = self.q_eval.forward(state)
            _, advantage = self.q_eval.forward(state)
            action = torch.argmax(advantage).item()
            return action
        return np.random.choice(np.arange(self.a_dim, dtype=np.int32))

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def learn(self):
        """Updates the network parameters based on sample batch from experience replay."""
        if self.replay_buffer.index < self.batch_size:
            return

        self.q_eval_optimizer.zero_grad()
        self.q_next_optimizer.zero_grad()

        self.replace_target_network()

        (
            state_batch,
            action_batch,
            reward_batch,
            new_state_batch,
            terminal_batch,
        ) = self.replay_buffer.sample_batch(self.batch_size)
        indices = np.arange(self.batch_size)

        v, a = self.q_eval.forward(state_batch)
        v_next, a_next = self.q_next.forward(new_state_batch)

        v_eval, a_eval = self.q_eval.forward(new_state_batch)

        q_pred = torch.add(v, (a - a.mean(dim=1, keepdim=True)))[indices, action_batch]
        q_next = torch.add(v_next, (a_next - a_next.mean(dim=1, keepdim=True)))
        q_eval = torch.add(v_eval, (a_eval - a_eval.mean(dim=1, keepdim=True)))

        max_actions = torch.argmax(q_eval, dim=1)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * q_next[indices, max_actions]

        loss = self.q_eval.loss_fn(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.loss = loss
        self.q_eval_optimizer.step()
        self.learn_step_counter += 1
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
        chkpt_dir_path = os.path.join(root_directory, "checkpoints", "ddqn", chkpt_name)
        os.makedirs(chkpt_dir_path, exist_ok=True)

        q_eval_path = os.path.join(chkpt_dir_path, "q_eval.pth")
        q_next_path = os.path.join(chkpt_dir_path, "q_next.pth")

        torch.save(self.q_eval, q_eval_path)
        torch.save(self.q_next, q_next_path)
        return chkpt_dir_path

    def __str__(self) -> str:
        return f"""
        DQNAgent params:
        \t epsilon: {self.epsilon}
        \t QEval Network: {self.q_eval}
        \t QNext Network: {self.q_next}
        """
