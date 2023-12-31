"""
Utility functions for training and evaluation.
"""
import os
import logging
from logging import Logger
from datetime import datetime
from pick import pick
import yaml
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from agents.dqn.qnetwork import QNetwork


def save_plots(rewards: list, losses: list):
    """Save training artefacts: loss and reward line plots.

    Parameters
    ----------
    rewards : list
        List of accumulated rewards for each episode over the entire run.
    losses : list
        List of QNetwork losses over the entire run.
    """
    plt.plot(rewards, label="Rewards")
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")

    z = np.polyfit(range(len(rewards)), rewards, 1)
    p = np.poly1d(z)
    plt.plot(range(len(rewards)), p(range(len(rewards))), "r--", label="Trendline")

    plt.legend()
    plt.savefig("artefacts/rewards_plot.png")
    plt.close()

    plt.plot(losses, label="Losses")
    plt.title("Training Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")

    z = np.polyfit(range(len(losses)), losses, 1)
    p = np.poly1d(z)
    plt.plot(range(len(losses)), p(range(len(losses))), "r--", label="Trendline")

    plt.legend()
    plt.savefig("artefacts/loss_plot.png")
    plt.close()


def save_checkpoint(network: QNetwork, iteration: int):
    """Saves QNetwork checkpoint at a specific iteration.

    Parameters
    ----------
    network : QNetwork
        QNetwork in training.
    iteration : int
        Iteration number at each network weights will be saved.
    """
    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    chkpt_name = f"qnetwork-{iteration}_eps-{timestamp}.pth"
    chkpt_path = os.path.join("checkpoints", "qnetwork", chkpt_name)
    torch.save(network, chkpt_path)


def capture_metrics(
    writer: SummaryWriter,
    rewards: list,
    reward: float,
    losses,
    loss: float,
    eps_history: list,
    epsilon: float,
) -> tuple[list, list, list]:
    """Stores loss, reward and epsilon along the run.

    Parameters
    ----------
    writer : SummaryWriter
        Tensorboard writer.
    rewards : list
        List of accumulated rewards
    reward : float
        Immediate reward from action in a state.
    losses : _type_
        List of QNetwrok losses.
    loss : float
        Loss of QNetwork.
    eps_history : list
        List of epsilon values along the run.
    epsilon : float
        Epsilon value for agent.

    Returns
    -------
    tuple[list, list, list]
        updated rewards losses and epis_history lists.
    """
    writer.add_scalar("Reward", reward)
    writer.add_scalar("Loss", loss)
    rewards.append(reward)
    losses.append(loss)
    eps_history.append(epsilon)
    return (rewards, losses, eps_history)


def load_hyperparamters(yaml_file: str) -> tuple[dict, dict, dict]:
    """Loads the hyperparameters from the YAML file.

    Parameters
    ----------
    yaml_file : str
        YAML file path.

    Returns
    -------
    tuple[dict, dict, dict]
        loaded hyperaparameters for agent, training loop and evaluation.
    """
    with open(yaml_file, "r", encoding="UTF-8") as file:
        hyperparameters = yaml.safe_load(file)
    agent_params = hyperparameters.get("agent", {})
    training_params = hyperparameters.get("training", {})
    eval_params = hyperparameters.get("evaluation", {})
    return agent_params, training_params, eval_params


def init_logger(logger_name: str) -> Logger:
    """Initializes module logger.

    Parameters
    ----------
    logger_name : str
        Logger name.

    Returns
    -------
    Logger
        Initiliazed logger.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(logger_name)
    return logger


def checkpoint_selection(checkpoint_path: str) -> str:
    """Display menu to select saved checkpoints.

    Parameters
    ----------
    checkpoint_path : str
        Checkpoint directory path

    Returns
    -------
    str
        checkpoint path
    """
    title = "Please choose the QNetwrok checkpoint: "
    options = [op for op in os.listdir(checkpoint_path) if op.endswith(".pth")]
    option, _ = pick(options, title, indicator=">", default_index=0)
    return os.path.join(checkpoint_path, option)
