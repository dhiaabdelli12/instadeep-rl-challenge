"""
Utility functions.
"""
import os
import logging
import yaml
from pick import pick
import matplotlib.pyplot as plt
import numpy as np


def load_hyperparamters(yaml_file: str, module: list | str):
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

    if isinstance(module, str):
        return hyperparameters.get(module, {})
    elif isinstance(module, list):
        results = []
        for m in module:
            results.append(hyperparameters.get(m, {}))
        return tuple(results)
    return None


def init_logger(logger_name: str):
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
    options = [op for op in os.listdir(checkpoint_path)]
    option, _ = pick(options, title, indicator=">>", default_index=0)
    return os.path.join(checkpoint_path, option)


def save_plots(chkpt_name: str, rewards: list, losses: list, fuel: list):
    """Save training artefacts: loss and reward line plots.

    Parameters
    ----------
    chkpt_name:
        Name of saved checkpoint.
    rewards : list
        List of accumulated rewards for each episode over the entire run.
    losses : list
        List of QNetwork losses over the entire run.
    """
    plt.plot(rewards, label="Rewards")
    plt.plot(fuel, label="Fuel Consumption")
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")

    z_reward = np.polyfit(range(len(rewards)), rewards, 1)
    p_reward = np.poly1d(z_reward)
    plt.plot(
        range(len(rewards)), p_reward(range(len(rewards))), "r--", label="Trendline"
    )

    z_fuel = np.polyfit(range(len(fuel)), fuel, 1)
    p_fuel = np.poly1d(z_fuel)
    plt.plot(range(len(fuel)), p_fuel(range(len(fuel))), "g--", label="Trendline")

    plt.legend()
    path = os.path.join(chkpt_name, "rewards_plot.png")
    plt.savefig(path)
    plt.close()

    plt.plot(losses, label="Losses")
    plt.title("Training Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")

    z = np.polyfit(range(len(losses)), losses, 1)
    p = np.poly1d(z)
    plt.plot(range(len(losses)), p(range(len(losses))), "r--", label="Trendline")

    plt.legend()
    path = os.path.join(chkpt_name, "loss_plot.png")
    plt.savefig(path)
    plt.close()
