"""
Evaluation script for Agent on the LunarLanding environment.
"""
import os
import sys
import gym
from utils import load_hyperparamters, init_logger, checkpoint_selection
from agents.dqn.agent import DQNAgent
from agents.ddqn.agent import DDQNAgent
from agents.ddpg.agent import DDPGAgent


agent_params, eval_params = load_hyperparamters(
    "config.yml", module=["agent", "evaluation"]
)


logger = init_logger("Evaluation")

if __name__ == "__main__":
    agent_choice = sys.argv[1]

    if agent_choice == "dqn":
        env = gym.make("LunarLander-v2", render_mode="human")
        path = os.path.join("checkpoints", "dqn")
        checkpoint_path = checkpoint_selection(path)
        agent = DQNAgent(
            env=env, name="dqn", chkpt_path=checkpoint_path, **agent_params
        )
        agent.epsilon = agent_params["epsilon_end"]
    elif agent_choice == "ddqn":
        env = gym.make("LunarLander-v2", render_mode="human")
        path = os.path.join("checkpoints", "ddqn")
        checkpoint_path = checkpoint_selection(path)
        agent = DDQNAgent(
            env=env, name="ddqn", chkpt_path=checkpoint_path, **agent_params
        )
    elif agent_choice == "ddpg":
        env = gym.make("LunarLander-v2", render_mode="human", continuous=True)
        path = os.path.join("checkpoints", "ddpg")
        checkpoint_path = checkpoint_selection(path)
        agent = DDPGAgent(
            env=env, name="ddpg", chkpt_path=checkpoint_path, **agent_params
        )

    for _ in range(eval_params["n_episodes"]):
        cumulative_reward = 0
        state, _ = env.reset()
        done = False

        while not done:
            action = agent.act(state)
            print(agent.epsilon)
            next_state, reward, done, _, _ = env.step(action)
            cumulative_reward += reward
            state = next_state
            logger.info("Reward: %.2f", cumulative_reward)
        logger.info("Episode ended with score: %.2f", cumulative_reward)

    env.close()
