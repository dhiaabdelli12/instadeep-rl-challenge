"""
RL agent base class implementation.
"""
import os
import json
import logging
from pathlib import Path
from datetime import datetime
import torch
import gym
from agents.buffer import ReplayBuffer


class RLAgent:
    """Base class from which all RL agents inherit"""

    def __init__(
        self, name: str, env: gym.Env, chkpt_path: None | str = None, **kwargs
    ) -> None:
        """
        Initialize the RLAgent.

        Parameters
        ----------
        name : str
            Name of the RLAgent.
        env : gym.Env
            Environment for the RLAgent.
        chkpt_path : str or None, optional
            Path to load checkpoint, by default None.
        **kwargs
            Additional parameters to be set as attributes of the RLAgent parsed from the YAML file.
        """
        self.params = kwargs
        self.__dict__.update(kwargs)
        self.s_dim = env.observation_space.shape[0]
        self.networks = {}
        self.replay_buffer = ReplayBuffer(self.mem_size, self.s_dim, self.device)
        self.chkpt_path = chkpt_path
        self.env = env
        self.name = name
        self.chkpt_dir = os.path.join(Path(__file__).parent.parent, "checkpoints", name)

        if chkpt_path:
            self.epsilon = self.epsilon_end
            self._load_networks(chkpt_path)

    def _load_networks(self, chkpt_path) -> None:
        """
        Load network checkpoints.
        """
        network_names = [n for n in os.listdir(chkpt_path) if n.endswith(".pth")]
        for network_name in network_names:
            self.networks[network_name.replace(".pth", "")] = torch.load(
                os.path.join(chkpt_path, network_name)
            )

    def save_networks(self, episodes: int, agent_params, training_params) -> str:
        """
        Save network checkpoints.

        Parameters
        ----------
        episodes : int
            Number of episodes.

        Returns
        -------
        str
            Path to the saved checkpoint directory.
        """

        timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M")
        name = f"{episodes}_eps-{timestamp}"
        output_path = os.path.join(self.chkpt_dir, name)
        os.makedirs(output_path, exist_ok=True)
        with open(
            os.path.join(output_path, "agent_params.json"), "w", encoding="UTF-8"
        ) as json_file:
            json.dump(agent_params, json_file, indent=4)
        with open(
            os.path.join(output_path, "training_params.json"), "w", encoding="UTF-8"
        ) as json_file:
            json.dump(training_params, json_file, indent=4)

        for _, network in self.networks.items():
            network.save_checkpoint(output_path)
        return output_path

    def _init_logger(self, name: str) -> logging.Logger:
        """
        Initialize a logger with the specified name.

        Parameters
        ----------
        name : str
            Name of the logger.

        Returns
        -------
        logging.Logger
            Initialized logger.
        """
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(name)
        return logger

    def act(self):
        """
        Implements the agent's action selection strategy.

        Raises
        ------
        NotImplementedError
            Method not implemented.
        """
        raise NotImplementedError("Method no implemented.")

    def learn(self):
        """
        Implements the agent's learning algorithm.

        Raises
        ------
        NotImplementedError
            Method not implemented.
        """
        raise NotImplementedError("Method no implemented.")

    def __str__(self):
        """
        Returns a string representation of the RLAgent.

        Raises
        ------
        NotImplementedError
            Method not implemented.
        """
        raise NotImplementedError("Method no implemented.")
