###############################################################################
# pogo_stick_random.py
#
# A script for evaluating performance with a random agent
#
# Created: 03/09/2021
#   - Andrew Albright
#   - andrew.albright1@louisiana.edu
#
###############################################################################

import os
from pathlib import Path
import numpy as np 

from stable_baselines3.common.monitor import Monitor

from gym_pogo_stick.gym_pogo_stick.envs import PogoStickControlEnv

# create path to save data
save_path = Path.cwd() / "Random_Action_Results"
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Env parameters
NUM_JUMPS = 2
EP_STEPS = 400
EPISODES = 1
REWARD_FUNCTIONS = ["Height", "Efficiency", "SpecHei", "SpHeEf"]

def random_action(reward_func):
    # Set up training env      
    env = PogoStickControlEnv(numJumps=NUM_JUMPS,
                              rewardType=reward_func,
                              linear="Nonlinear",
                              epSteps=EP_STEPS, 
                              evaluating=False)

    obs = env.reset()
    steps = EP_STEPS * EPISODES
    for _ in range(steps):
      action = env.action_space.sample()
      obs, reward, done, info = env.step(action)
      env.render()
      if done:
        env.reset()
        env.pogo_stick.modify_design()
    env.close()

if __name__ == "__main__":
  for rew in REWARD_FUNCTIONS:
    random_action(rew)