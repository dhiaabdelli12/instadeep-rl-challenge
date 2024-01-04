"""
Implementation of a heuristic-based agent.
"""
import logging
import numpy as np
from gym.wrappers.time_limit import TimeLimit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HeuristicAgent:
    """Implements the heuristic agent."""

    def __init__(self, env: TimeLimit) -> None:
        self.s_dim = env.observation_space.shape[0]
        self.a_dim = env.action_space.n

    def act(state: np.ndarray) -> int:
        ...

    def __str__(self) -> str:
        return f"""
        Agent rules:
        \t Rule 1: 
        \t Rule 2: 
        """
