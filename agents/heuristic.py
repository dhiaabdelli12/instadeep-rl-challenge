"""
Implementation of the heuristic method.
"""
import gym
import numpy as np
from time import sleep


class Heurstic:
    def __init__(self, env: gym.Env) -> None:
        """Initiliazes the heurisitc class.

        Parameters
        ----------
        env : gym.Env
            Gym environment
        """
        self.env = env

    def act(self, state: np.ndarray) -> int:
        """Returns action based on set of rules.

        Parameters
        ----------
        state : np.ndarray
            Current state.

        Returns
        -------
        int
            Chosen action.
        """
        main_engine_treshold = 0.8
        pad_bounds = (-0.2, 0.2)
        angle_bounds = (-0.2, 0.2)

        x, y, dx, dy, angle, angl_vel, l_ground, r_ground = state
        if angle > angle_bounds[1]:
            return 3
        if angle < angle_bounds[0]:
            return 1

        if (
            x < pad_bounds[1] and x > pad_bounds[0]
        ):  # ship is apporximately within the landing pad bounds
            if l_ground or r_ground:  # one of the legs touched ground
                return 0
            if y < main_engine_treshold:
                return 2
        elif x <= pad_bounds[0]:  # ship is left to pad
            return 1
        elif x >= pad_bounds[1]:  # ship is right to pad
            return 3

        return 0
