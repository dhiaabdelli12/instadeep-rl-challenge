"""
Implementation of the Deep Deterministic Policy Gradient agent.
"""
import numpy as np
import torch
import gym
import torch.nn.functional as F
from agents.ddpg.network import Actor, Critic
from agents.rlagent import RLAgent
from agents.buffer import ContinuousReplayBuffer


class OUActionNoise(object):
    """
    Ornstein-Uhlenbeck Action Noise for exploration in continuous action spaces.
    """

    def __init__(
        self,
        mu: float,
        sigma: float = 0.15,
        theta: float = 0.2,
        dt: float = 1e-2,
        x0: float | None = None,
    ) -> None:
        """Initialize the OUActionNoise.

        Parameters
        ----------
        mu : float
            Mean around which the noise fluctuates.
        sigma : float, optional
            Volatility or amplitude of the noise., by default 0.15
        theta : float, optional
            Rate of mean reversion., by default 0.2
        dt : float, optional
            Time step., by default 1e-2
        x0 : float | None, optional
            Initial state, if None, defaults to zero.
        """
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        """
        Generate and return the next noise value.

        Returns:
            numpy.ndarray: Noise sample.
        """
        x = (
            self.x_prev
            + self.theta * (self.mu - self.x_prev) * self.dt
            + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        )
        self.x_prev = x
        return x

    def reset(self):
        """
        Reset the noise process to its initial state.

        Returns:
            None
        """
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        """
        Return a string representation of the OUActionNoise.

        Returns:
            str: String representation.
        """
        return "OrnsteinUhlenbeckActionNoise(mu={}, sigma={})".format(
            self.mu, self.sigma
        )


class DDPGAgent(RLAgent):
    """
    Deep Deterministic Policy Gradient implementation class.
    """

    def __init__(
        self, name: str, env: gym.Env, chkpt_path: str | None = None, **kwargs
    ) -> None:
        """
        Initialize the DDPGAgent.

        Parameters
        ----------
        name : str
            Name of the DDPGAgent.
        env : gym.Env
            Environment for the DDPGAgent.
        chkpt_path : str or None, optional
            Path to load checkpoint, by default None.
        **kwargs
            Additional parameters to be set as attributes of the DDPGAgent.
        """
        super().__init__(name, env, chkpt_path, **kwargs)
        self.replay_buffer = ContinuousReplayBuffer(
            self.mem_size, self.s_dim, self.device
        )
        self.a_dim = 2
        if not chkpt_path:
            self.networks = {
                "actor": Actor(
                    name="actor",
                    lr=self.alpha,
                    input_dim=self.s_dim,
                    fc1_dim=400,
                    fc2_dim=300,
                    output_dim=0,
                    n_action=self.a_dim,
                    device=self.device,
                ),
                "t_actor": Actor(
                    name="t_actor",
                    lr=self.alpha,
                    input_dim=self.s_dim,
                    fc1_dim=400,
                    fc2_dim=300,
                    output_dim=0,
                    n_action=self.a_dim,
                    device=self.device,
                ),
                "critic": Critic(
                    name="critic",
                    lr=self.alpha,
                    input_dim=self.s_dim,
                    fc1_dim=400,
                    fc2_dim=300,
                    output_dim=0,
                    n_action=self.a_dim,
                    device=self.device,
                ),
                "t_critic": Critic(
                    name="t_critic",
                    lr=self.alpha,
                    input_dim=self.s_dim,
                    fc1_dim=400,
                    fc2_dim=300,
                    output_dim=0,
                    n_action=self.a_dim,
                    device=self.device,
                ),
            }
        self.noise = OUActionNoise(mu=np.zeros(self.a_dim))
        self.update_network_parameters(tau=1)

    def act(self, state):
        """
        Select an action.

        Parameters
        ----------
        state : np.ndarray
            Current state.

        Returns
        -------
        int
            Selected action.
        """
        self.networks["actor"].eval()
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        mu = self.networks["actor"](state).to(self.device)
        mu_prime = mu + torch.tensor(self.noise(), dtype=float).to(self.device)

        self.networks["actor"].train()
        return mu_prime.cpu().detach().numpy()

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

        self.networks["t_actor"].eval()
        self.networks["t_critic"].eval()
        self.networks["critic"].eval()

        target_actions = self.networks["t_actor"](new_state_batch)
        critic_value_ = self.networks["t_critic"](new_state_batch, target_actions)
        critic_value = self.networks["critic"](state_batch, action_batch)

        target = []
        for j in range(self.batch_size):
            target.append(
                reward_batch[j] + self.gamma * critic_value_[j] * terminal_batch[j]
            )
        target = torch.tensor(target).to(self.device)
        target = target.view(self.batch_size, 1)

        self.networks["critic"].train()
        self.networks["critic"].optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.networks["critic"].optimizer.step()

        self.networks["critic"].eval()
        self.networks["actor"].optimizer.zero_grad()
        mu = self.networks["actor"](state_batch)
        self.networks["actor"].train()
        actor_loss = -self.networks["critic"](state_batch, mu)
        actor_loss = torch.mean(actor_loss)

        self.loss = torch.Tensor.cpu(actor_loss.detach()).numpy()

        actor_loss.backward()
        self.networks["actor"].optimizer.step()
        self.update_network_parameters()

    def update_network_parameters(self, tau: float = None):
        """
        Update the parameters of the target networks using a soft update.

        Parameters
        ----------
        tau : float, optional
            Soft update coefficient, by default None.

        Returns
        -------
        None
        """
        if tau is None:
            tau = self.tau
        actor_params = self.networks["actor"].named_parameters()
        critic_params = self.networks["critic"].named_parameters()
        target_actor_params = self.networks["t_actor"].named_parameters()
        target_critic_params = self.networks["t_critic"].named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)

        target_critic_dict = dict(target_critic_params)
        target_actor_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = (
                tau * critic_state_dict[name].clone()
                + (1 - tau) * target_critic_dict[name].clone()
            )

        self.networks["t_critic"].load_state_dict(critic_state_dict)

        for name in actor_state_dict:
            actor_state_dict[name] = (
                tau * actor_state_dict[name].clone()
                + (1 - tau) * target_actor_dict[name].clone()
            )

        self.networks["t_actor"].load_state_dict(actor_state_dict)
