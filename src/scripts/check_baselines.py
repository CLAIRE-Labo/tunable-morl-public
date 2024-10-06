# Adapted from https://colab.research.google.com/drive/1ByjuUp8-CJeh1giPOACqPGiglPxDnlSq?usp=sharing
# The accompanying notebook for MORL Baselines
import sys
from pathlib import Path
import numpy as np
import mo_gymnasium as mo_gym
from mo_gymnasium.utils import MORecordEpisodeStatistics
from morl_baselines.multi_policy.pareto_q_learning.pql import PQL
import wandb


GAMMA = 0.99

env = mo_gym.make("deep-sea-treasure-v0")
env = MORecordEpisodeStatistics(env, gamma=GAMMA)  # wrapper for recording statistics

eval_env = mo_gym.make("deep-sea-treasure-concave-v0") # environment used for evaluation

agent = PQL(
    env=env,
    ref_point=np.array([0, -50]),  # used to compute hypervolume
    gamma=GAMMA,
    log=True,  # use weights and biases to see the results!
)
agent.setup_wandb(project_name="tunable-morl", experiment_name="PQL")
agent.train(total_timesteps=100000, eval_env=eval_env, ref_point=np.array([0, -50]))

print(env.pareto_front(0.9))
