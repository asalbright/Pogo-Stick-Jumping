###############################################################################
# zevaluation_SpringK.py
#
# A script for evaluating the model produced by pogo_stick_training
#
# Created: 02/23/2021
#   - Andrew Albright
#   - andrew.albright1@louisiana.edu
#
###############################################################################

import os
from pathlib import Path
import gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from functions import getZipFileName

import pogo_stick_jumping

# log paths
save_path = Path.cwd()
models_path = save_path.joinpath('models')
figures_data_path = save_path.joinpath('figures_data')

if not os.path.exists(models_path):
    os.makedirs(models_path)
if not os.path.exists(figures_data_path):
    os.makedirs(figures_data_path)

# Call function to get agent names from models_path folder
agent_names = getZipFileName(path=models_path)

env_id = 'pogo-stick-jumping-v11'       # Nonlinear Environment
EPISODE_SIMS = 60
SIM_STEP_SIZE = 0.001
SIM_DURATION = 1
REWARD_FUNCTION = 'RewardHeight'
spring_k = 5760
variance = 0.75
MIN_SPRING_K = spring_k - variance * spring_k
MAX_SPRING_K = spring_k + variance * spring_k 

for agent in agent_names:
    env = gym.make(env_id)
    env.init_variables(NUM_SIMS=EPISODE_SIMS,
                    SIM_STEP=SIM_STEP_SIZE,
                    SIM_DURATION=SIM_DURATION, 
                    REWARD_FUNCTION=REWARD_FUNCTION,  
                    MIN_SPRING_K=MIN_SPRING_K,
                    MAX_SPRING_K=MAX_SPRING_K,
                    SAVE_DATA=True,
                    SAVE_NAME=agent, 
                    SAVE_PATH=figures_data_path)
    # env = Monitor(gym.make(env_id))

    # Deploy the agent
    # load the agent
    test_agent = models_path / agent
    model = TD3.load(path=test_agent)

    done, state = False, False
    obs = env.reset()
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
    env.close()