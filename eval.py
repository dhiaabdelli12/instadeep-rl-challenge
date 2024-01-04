"""
Evaluation script for Agent on the LunarLanding environment.
"""
import os
import sys
import gymnasium as gym
from utils import load_hyperparamters, init_logger, checkpoint_selection
from agents.dqn.agent import DQNAgent
from agents.ddqn.agent import DDQNAgent


agent_params, eval_params = load_hyperparamters(
    "config.yml", module=["agent", "evaluation"]
)
logger = init_logger("Evaluation")

if __name__ == "__main__":
    env = gym.make("LunarLander-v2", render_mode="human")

    agent_choice = sys.argv[1]
    if agent_choice == "dqn":
        path = os.path.join("checkpoints", "dqn")
        checkpoint_path = checkpoint_selection(path)
        agent = DQNAgent(env=env, checkpoint_path=checkpoint_path, **agent_params)
    elif agent_choice == "ddqn":
        path = os.path.join("checkpoints", "ddqn")
        checkpoint_path = checkpoint_selection(path)
        agent = DDQNAgent(env=env, checkpoint_path=checkpoint_path, **agent_params)

    for _ in range(eval_params["n_episodes"]):
        SCORE = 0
        state, _ = env.reset()
        DONE = False

        while not DONE:
            action = agent.act(state)
            n_state, reward, DONE, _, _ = env.step(action)
            SCORE += reward
            state = n_state
            logger.info("Score: %.2f", SCORE)
        logger.info("Episode ended with score: %.2f", SCORE)
    env.close()
