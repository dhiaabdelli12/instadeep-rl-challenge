"""
Training script for Agent on the LunarLanding environment.
"""
import os
from datetime import datetime
import logging
import warnings

import yaml
import gym
import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from agents.dqn.agent import DQNAgent
from utils import save_plots

warnings.filterwarnings("ignore", category=DeprecationWarning)
writer = SummaryWriter()
with open("config.yml", "r", encoding="UTF-8") as file:
    hyperparameters = yaml.safe_load(file)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Agent")


if __name__ == "__main__":
    agent_params = hyperparameters.get("agent", {})
    training_params = hyperparameters.get("training", {})

    env = gym.make("LunarLander-v2")
    agent = DQNAgent(env=env, **agent_params)

    logger.info("Training on: %s", agent.qnetwork.device)

    n_episodes = training_params["n_episodes"]
    scores, eps_history = [], []
    losses = []

    for i in range(n_episodes + 1):
        SCORE = 0
        state, _ = env.reset()
        DONE = False

        while not DONE:
            action = agent.act(state)
            n_state, reward, DONE, _, _ = env.step(action)
            SCORE += reward
            agent.store_transition(state, action, reward, n_state, DONE)
            agent.learn()
            state = n_state

        writer.add_scalar("Reward", SCORE)
        writer.add_scalar("Loss", Tensor.cpu(agent.qnetwork.loss.detach()).numpy())
        scores.append(SCORE)
        losses.append(Tensor.cpu(agent.qnetwork.loss.detach()).numpy())

        eps_history.append(agent.epsilon)
        if i % training_params["eval_interval"] == 0:
            logger.info(
                "[Episode %d/%d]: Reward: %.2f\t Loss: %.2f\tEpsilon: %.2f",
                i,
                n_episodes,
                SCORE,
                agent.qnetwork.loss,
                agent.epsilon,
            )

        if i % training_params["checkpoint_interval"] == 0 and i != 0:
            logger.info("Saving QNetwork checkpoint")
            timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            chkpt_name = f"qnetwork-{i}_eps-{timestamp}.pth"
            chkpt_path = os.path.join("checkpoints", "qnetwork", chkpt_name)
            torch.save(agent.qnetwork, chkpt_path)

    save_plots(scores, losses)

    env.close()
