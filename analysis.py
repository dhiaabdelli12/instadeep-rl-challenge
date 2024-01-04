"""
Interpretability analysis for the trained agent checkpoints.
"""

import os
import warnings
import gym
import numpy as np
import pandas as pd
from time import time
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from agents.dqn.agent_ import DQNAgent
from utils import load_hyperparamters, checkpoint_selection, init_logger

warnings.simplefilter("ignore")

path = os.path.join("checkpoints", "dqn")
checkpoint_path = checkpoint_selection(path)
agent_params, analysis_params = load_hyperparamters(
    "config.yml", module=["agent", "analysis"]
)
logger = init_logger("Analysis")


if __name__ == "__main__":
    env = gym.make("LunarLander-v2")
    agent = DQNAgent(env=env, checkpoint_path=checkpoint_path, **agent_params)

    i = 0
    columns = [
        "x",
        "y",
        "dx",
        "dy",
        "angle",
        "ang_vel",
        "l_ground",
        "l_ground",
        "action",
    ]
    samples = pd.DataFrame(columns=columns)

    while i < analysis_params["n_samples"]:
        state, _ = env.reset()
        DONE = False
        if not DONE:
            action = agent.act(state)
            logger.info("Collecting samples: %i/%i", i, analysis_params["n_samples"])
            new_row = np.append(state, action)
            samples.loc[len(samples)] = new_row
            i += 1
            n_state, reward, DONE, _, _ = env.step(action)
            state = n_state

    output_path = os.path.join(
        "checkpoints", "dqn", agent.checkpoint_name, "samples.csv"
    )
    samples.to_csv(output_path)

    logger.info(f"Data saved to {output_path}")

    X = samples.drop(columns=["action"])
    y = samples[["action"]]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=analysis_params["test_size"], random_state=42
    )

    dt = DecisionTreeClassifier(random_state=42)
    logger.info("Fitting decision tree model")
    now = time()
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    then = time()

    tree_rules = export_text(dt, feature_names=list(X.columns))

    analysis_path = os.path.join(
        "checkpoints", "dqn", agent.checkpoint_name, "analysis.txt"
    )

    with open(analysis_path, "a", encoding="UTF-8") as file:
        file.write(
            f"""
        Time elapsed:{(then-now):.2f}
        accuracy: {accuracy}
        precision: {precision}
        recall: {recall}
        f1: {f1}
        """
        )

        file.write(tree_rules)

    env.close()
