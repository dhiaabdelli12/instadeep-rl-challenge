"""
Training script for Agent on the LunarLanding environment.
"""
import warnings
import sys
import gym
from utils import load_hyperparamters, init_logger, save_plots
from agents.dqn.agent import DQNAgent
from agents.ddqn.agent import DDQNAgent
from agents.ddpg.agent import DDPGAgent

warnings.simplefilter("ignore")

agent_params, training_params = load_hyperparamters(
    "config.yml", module=["agent", "training"]
)

logger = init_logger("Training")


if __name__ == "__main__":
    agent_choice = sys.argv[1]

    if agent_choice == "dqn":
        env = gym.make("LunarLander-v2")
        agent = DQNAgent(name="dqn", env=env, **agent_params)
    elif agent_choice == "ddqn":
        env = gym.make("LunarLander-v2")
        agent = DDQNAgent(name="ddqn", env=env, **agent_params)
    elif agent_choice == "ddpg":
        env = gym.make("LunarLander-v2", continuous=True)
        agent = DDPGAgent("ddpg", env=env, **agent_params)
    else:
        logger.error("No agent with that name.")
        exit()

    logger.info("Training on %s", agent.device)

    rewards, losses = [], []
    eps_history, fuel_consumption = [], []

    for i in range(training_params["n_episodes"] + 1):
        cumulative_reward = 0
        state, _ = env.reset()
        done = False
        steps_per_episode = 0
        fuel = 0

        while not done and steps_per_episode < training_params["max_steps_per_episode"]:
            steps_per_episode += 1
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)

            if training_params["fuel_penalty"] > 1:
                if agent_choice in ["dqn", "ddqn"]:
                    if action in [1, 3]:
                        reward = -0.03 * training_params["fuel_penalty"]
                        fuel += 0.03
                    elif action == 2:
                        reward = -0.3 * training_params["fuel_penalty"]
                        fuel += 0.3

            cumulative_reward += reward
            agent.replay_buffer.store_transition(
                state, action, reward, next_state, done
            )
            agent.learn()
            state = next_state

        if agent_choice == "dqn":
            loss = agent.networks["qnetwork"].loss
        elif agent_choice == "ddqn":
            loss = agent.networks["q_eval"].loss
        elif agent_choice == "ddpg":
            loss = agent.loss
        losses.append(loss)
        fuel_consumption.append(fuel)
        rewards.append(cumulative_reward)

        epsilon = agent.epsilon if agent_choice in ["dqn", "ddqn"] else 0

        if i % training_params["log_interval"] == 0 and i != 0:
            logger.info(
                "[Episode %d/%d]: Reward: %.2f\tLoss: %.2f\tEpsilon: %.2f\t Updates: %i",
                i,
                training_params["n_episodes"],
                cumulative_reward,
                loss,
                epsilon,
                steps_per_episode / agent_params["batch_size"],
            )

        if i % training_params["checkpoint_interval"] == 0 and i != 0:
            path = agent.save_networks(i, agent_params, training_params)
            logger.info("Checkpoints saved.")
            save_plots(path, rewards, losses, fuel_consumption)

    env.close()
