"""
Evaluation script for Agent on the LunarLanding environment.
"""
import os
import yaml

import gymnasium as gym
from agents.dqn.agent import DQNAgent


env = gym.make("LunarLander-v2", render_mode="human")
path = os.path.join(
    "checkpoints", "qnetwork", "qnetwork-700_eps-2023-12-30T18:49:20.pth"
)

with open("config.yml", "r", encoding="UTF-8") as file:
    hyperparameters = yaml.safe_load(file)


agent_params = hyperparameters.get("agent", {})
eval_params = hyperparameters.get("evaluation", {})
agent = DQNAgent(env=env, checkpoint_path=path, **agent_params)


if __name__ == "__main__":
    for _ in range(eval_params["n_episodes"]):
        SCORE = 0
        state, _ = env.reset()
        DONE = False

        while not done:
            action = env.action_space.sample()
            n_state, reward, done, _, _ = env.step(action)
            SCORE += reward
            state = n_state
            print(f"score: {SCORE}")
