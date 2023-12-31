"""
Evaluation script for Agent on the LunarLanding environment.
"""
import os
import gymnasium as gym
from utils import load_hyperparamters, init_logger, checkpoint_selection
from agents.dqn.agent import DQNAgent


path = os.path.join("checkpoints", "qnetwork")
checkpoint_path = checkpoint_selection(path)
agent_params, _, eval_params = load_hyperparamters("config.yml")
logger = init_logger("Evaluation")

if __name__ == "__main__":
    env = gym.make("LunarLander-v2", render_mode="human")
    agent = DQNAgent(env=env, checkpoint_path=checkpoint_path, **agent_params)
    for _ in range(eval_params["n_episodes"]):
        SCORE = 0
        state, _ = env.reset()
        DONE = False

        while not DONE:
            action = env.action_space.sample()
            n_state, reward, DONE, _, _ = env.step(action)
            SCORE += reward
            state = n_state
            logger.info("Score: %.2f", SCORE)
    env.close()
