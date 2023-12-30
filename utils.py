"""
Utility functions for training and evaluation.
"""

import matplotlib.pyplot as plt
import numpy as np


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