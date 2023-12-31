"""
Training script for Agent on the LunarLanding environment.
"""
import warnings
import gym
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from agents.dqn.agent import DQNAgent
from utils import (
    save_plots,
    save_checkpoint,
    capture_metrics,
    load_hyperparamters,
    init_logger,
)

warnings.simplefilter("ignore")
writer = SummaryWriter()
agent_params, training_params, _ = load_hyperparamters("config.yml")
logger = init_logger("Training")


if __name__ == "__main__":
    env = gym.make("LunarLander-v2")
    agent = DQNAgent(env=env, **agent_params)

    logger.info("Training on: %s", agent.qnetwork.device)

    scores, eps_history = [], []
    losses = []

    for i in range(training_params["n_episodes"] + 1):
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

        scores, losses, eps_history = capture_metrics(
            writer,
            scores,
            SCORE,
            losses,
            Tensor.cpu(agent.qnetwork.loss.detach()).numpy(),
            eps_history,
            agent.epsilon,
        )

        if i % training_params["eval_interval"] == 0:
            logger.info(
                "[Episode %d/%d]: Reward: %.2f\t Loss: %.2f\tEpsilon: %.2f",
                i,
                training_params["n_episodes"],
                SCORE,
                agent.qnetwork.loss,
                agent.epsilon,
            )

        if i % training_params["checkpoint_interval"] == 0 and i != 0:
            logger.info("Saving QNetwork checkpoint")
            save_checkpoint(agent.qnetwork, i)

    save_plots(scores, losses)
    env.close()
