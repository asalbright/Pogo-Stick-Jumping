###############################################################################
# training.py
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
import torch
from stable_baselines3 import TD3, PPO
from stable_baselines3.common.monitor import Monitor

from gym_pogo_stick.gym_pogo_stick.envs import PogoStickControlEnv

# Make sure a GPU is available for torch to utilize
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(torch.cuda.current_device()))
print(torch.cuda.is_available())

# Set Environment parameters
ENV_TYPE = "Nonlinear"
EP_STEPS = 400
# "OneJump" , "StutterJump" , "TimeJump"
JUMP_TYPE = "StutterJump"   
# Determine the number of jumps to send to the env
if JUMP_TYPE == "OneJump":
    num_jumps = 1
elif JUMP_TYPE == "StutterJump":
    num_jumps = 2

# Set up the Training parameters
NUM_TRIALS = 50
N_TRAINING_STEPS = 500000
ROLLOUT = 5000
GAMMA = 0.99

# Set up the training seeds
INITIAL_SEED = 70504
EVALUATION_FREQ = 25000
RNG = np.random.default_rng(INITIAL_SEED)
TRIAL_SEEDS = RNG.integers(low=0, high=1000, size=(NUM_TRIALS))
NUM_CPUs = 10

# Define the Reward Function to be used
# "Height", "Efficiency"
REWARD_FUNCTIONS = ["Height", "Efficiency", "SpecHei"]
ROBUST = False

def train_agents(seed=12345):

    for REWARD_FUNCTION in REWARD_FUNCTIONS:
        # Set up the training save paths
        data_name = f'{REWARD_FUNCTION[0:5]}_{JUMP_TYPE}'
        save_path = Path.cwd()
        save_path = save_path.joinpath(f'trained_{data_name}')
        logs_path = save_path.joinpath('logs')
        models_path = save_path.joinpath('models')
        if not os.path.exists(logs_path):
            os.makedirs(logs_path)
        if not os.path.exists(models_path):
            os.makedirs(models_path)

        # Set up training env
        env = PogoStickControlEnv(numJumps=num_jumps,
                                  linear=ENV_TYPE,
                                  trainRobust=ROBUST,
                                  epSteps=EP_STEPS, 
                                  evaluating=False,
                                  rewardType=REWARD_FUNCTION,
                                  specifiedHeight=0.05,
                                  captureData=False)
        
        # set the trial seed for use during training
        trial_seed = int(seed)
        env.seed(seed=trial_seed)

        # wrap the env in modified monitor which plots to tensorboard the jumpheight
        env = Monitor(env)

        buffer_size = N_TRAINING_STEPS + 1
        model = TD3(policy="MlpPolicy", 
                    env=env, 
                    verbose=1, 
                    tensorboard_log=logs_path, 
                    buffer_size=buffer_size, 
                    learning_starts=ROLLOUT,
                    seed=trial_seed, 
                    gamma=GAMMA)

        # open tensorboard with the following bash command: tensorboard --logdir ./logs/
        # train the agent
        model.learn(total_timesteps=N_TRAINING_STEPS, 
                    tb_log_name=f'{data_name}_{int(seed)}')
                    
        # Save the model at the end                
        path = models_path / f'final_{data_name}_{int(seed)}'
        model.save(path=path)
        
        # close the environment
        env.close()

def run_multi_processing(function, i, n_processors):
    with Pool(processes=n_processors) as pool:
        return pool.map(function, i)

if __name__ == "__main__":
    start = time.time()
    run_multi_processing(train_agents, TRIAL_SEEDS, NUM_CPUs)
    end = time.time()
    total = end - start
    print(f'Total Time: {total}')
    # train_agents()