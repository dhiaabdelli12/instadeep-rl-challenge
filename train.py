"""
Training script for Agent on the LunarLanding environment.
"""
import warnings
import sys
import gym
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from agents.dqn.agent import DQNAgent
from agents.ddqn.agent import DDQNAgent
from utils import (
    save_plots,
    capture_metrics,
    load_hyperparamters,
    init_logger,
)

from time import sleep

warnings.simplefilter("ignore")
writer = SummaryWriter()
agent_params, training_params = load_hyperparamters(
    "config.yml", module=["agent", "training"]
)


logger = init_logger("Training")


if __name__ == "__main__":
    env = gym.make("LunarLander-v2")
    agent_choice = sys.argv[1]
    if agent_choice == "dqn":
        agent = DQNAgent(env=env, **agent_params)
    elif agent_choice == "ddqn":
        agent = DDQNAgent(env=env, **agent_params)

    logger.info("Training %s on: %s", agent_choice, agent.device)

    scores, eps_history = [], []
    losses, fuel_consumption = [], []
    steps_per_episode = 0
    for i in range(training_params["n_episodes"] + 1):
        SCORE = 0
        state, _ = env.reset()
        DONE = False
        steps_per_episode = 0
        fuel = 0

        while not DONE and steps_per_episode < training_params["max_steps_per_episode"]:
            steps_per_episode += 1
            action = agent.act(state)
            n_state, reward, DONE, _, _ = env.step(action)
            if action == 1 or action == 3:
                fuel += 0.03
                reward *= 3
            elif action == 2:
                fuel += 0.3
                reward *= 3

            SCORE += reward
            agent.replay_buffer.store_transition(state, action, reward, n_state, DONE)
            agent.learn()
            state = n_state

        if agent_choice == "dqn":
            loss = Tensor.cpu(agent.qnetwork.loss.detach()).numpy()
        elif agent_choice == "ddqn":
            loss = Tensor.cpu(agent.q_eval.loss.detach()).numpy()
        scores, losses, eps_history = capture_metrics(
            writer,
            scores,
            SCORE,
            losses,
            loss,
            eps_history,
            agent.epsilon,
        )
        fuel_consumption.append(fuel)


        if i % training_params["log_interval"] == 0 and i != 0:
            logger.info(
                "[Episode %d/%d]: Reward: %.2f\t Loss: %.2f\tEpsilon: %.2f\tUpdates made: %i",
                i,
                training_params["n_episodes"],
                SCORE,
                loss,
                agent.epsilon,
                steps_per_episode / agent_params["batch_size"],
            )

        if i % training_params["checkpoint_interval"] == 0 and i != 0:
            logger.info("Saving Checkpoint")
            chkpt_name = agent.save_checkpoint(i)
            save_plots(chkpt_name, scores, losses, fuel_consumption)

    env.close()
