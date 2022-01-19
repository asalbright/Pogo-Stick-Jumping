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

from pogo_stick_jumping.pogo_stick_jumping_springActionNonlinear import PogoJumpingEnv


EPISODE_STEPS = 5
SIM_STEP_SIZE = 0.001
SIM_DURATION = 1
REWARD_FUNCTION = 'RewardHeight'
SPRING_K = 5760
VARIANCE = 0.75
MIN_SPRING_K = SPRING_K - VARIANCE * SPRING_K
MAX_SPRING_K = SPRING_K + VARIANCE * SPRING_K

# Set up training env
env = PogoJumpingEnv(EVALUATING=False,
                     EP_STEPS=EPISODE_STEPS,
                     SIM_STEP=SIM_STEP_SIZE,
                     SIM_DURATION=SIM_DURATION, 
                     REWARD_FUNCTION=REWARD_FUNCTION,  
                     MIN_SPRING_K=MIN_SPRING_K,
                     MAX_SPRING_K=MAX_SPRING_K,
                     SAVE_DATA=False,
                     SAVE_NAME=None, 
                     SAVE_PATH=False)
env = Monitor(env)  

num_steps = 1000
obs = env.reset()
for _ in range(num_steps):
  action = env.action_space.sample()
  print(f"Spring K: {action}")
  obs, reward, done, info = env.step(action)
  if done:
    env.reset()
env.close()