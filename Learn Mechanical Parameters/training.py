###############################################################################
# training_linear.py
#
# A script for training the pogo_sick env varying the weights on the reward
# 
#
# Created: 03/29/2021
#   - Andrew Albright
#   - andrew.albright1@louisiana.edu
#
###############################################################################

import os
from pathlib import Path
import time
import numpy as np 
from torch.multiprocessing import Pool, Process, set_start_method
try:
     set_start_method('spawn')
except RuntimeError:
    pass
import gym
import torch
import stable_baselines3
from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor 
from stable_baselines3.common.env_util import make_vec_env

from pogo_stick_jumping.pogo_stick_jumping_springActionNonlinear import PogoJumpingEnv
from custom_callbacks import LogMechParamsCallback

# Make sure a GPU is available for torch to utilize
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(torch.cuda.current_device()))
print(torch.cuda.is_available())

# Define the environment ID's and set Environment parameters
env_id = 'pogo-stick-jumping-v11'       # Nonlinear Environment
eval_env_id = 'pogo-stick-jumping-v11'
EPISODE_STEPS = 1
SIM_STEP_SIZE = 0.001
SIM_DURATION = 1
REWARD_FUNCTION = 'RewardHeight'
SPRING_K = 5760
VARIANCE = 1
MIN_SPRING_K = SPRING_K - VARIANCE * SPRING_K
MAX_SPRING_K = SPRING_K + VARIANCE * SPRING_K  
ZETA = 0.01
MIN_ZETA = ZETA - VARIANCE * ZETA
MAX_ZETA = ZETA + VARIANCE * ZETA

# Set up the Training parameters
NUM_TRIALS = 100
TOTAL_SIMS = 1000
ROLLOUT = 100
GAMMA = 0.99

# Set up the training seeds
INITIAL_SEED = 70504
RNG = np.random.default_rng(INITIAL_SEED)
TRIAL_SEEDS = RNG.integers(low=0, high=10000, size=(NUM_TRIALS))

# Set up the training save paths
data_name = f'{REWARD_FUNCTION}'
save_path = Path.cwd()
save_path = save_path.joinpath(f'train_{data_name}')
logs_path = save_path.joinpath('logs')
models_path = save_path.joinpath('models')
if not os.path.exists(logs_path):
    os.makedirs(logs_path)
if not os.path.exists(models_path):
    os.makedirs(models_path)

# Begin loop for testing different Omega_X values
def train_agents(seed=12345):

    # Set up training env
    env = PogoJumpingEnv(ep_steps=EPISODE_STEPS,
                         sim_step_size=SIM_STEP_SIZE,
                         sim_duration=SIM_DURATION, 
                         reward_function=REWARD_FUNCTION,  
                         min_spring_k=MIN_SPRING_K,
                         max_spring_k=MAX_SPRING_K,
                         min_zeta=MIN_ZETA,
                         max_zeta=MAX_ZETA)

    # set the trial seed for use during training
    trial_seed = int(seed)

    env.seed(seed=trial_seed)

    # wrap the env in modified monitor which plots to tensorboard the jumpheight
    env = Monitor(env)

    # create the model
    # open tensorboard with the following bash command: tensorboard --logdir ./logs/
    buffer_size = TOTAL_SIMS + 1
    model = TD3("MlpPolicy", 
                env, 
                verbose=1, 
                tensorboard_log=logs_path, 
                buffer_size=buffer_size, 
                learning_starts=ROLLOUT, 
                seed=trial_seed, 
                gamma=GAMMA)

    # train the agent
    callback = LogMechParamsCallback()
    model.learn(total_timesteps=TOTAL_SIMS, callback=callback, tb_log_name=f'{data_name}_{int(seed)}_log')

def multi_process(function, i, n_processors):
    with Pool(processes=n_processors) as pool:
        return pool.map(function, i)

if __name__ == "__main__":
    start = time.time()
    multi_process(train_agents, TRIAL_SEEDS, 3)
    end = time.time()
    total = end - start
    print(f'Total Time: {total}')
    # train_agents()
    