#! /usr/bin/env python

###############################################################################
# pogo_design_env.py
#
# Defines a pogo stick jumping environment for use with the openAI Gym.
# Intendted to find a design for a pogo stick that can be used to jump
#
# Created: 01/04/2022
#   - Andrew Albright
#   - andrew.albright1@louisiana.edu
#
# Notes:
###############################################################################

import os
import sys
import gym
from gym import spaces
from gym.utils import seeding
import logging
import numpy as np
from scipy.integrate import solve_ivp
import datetime # for unique filenames
from pathlib import Path
from stable_baselines3.common.logger import record
from gym_pogo_stick.gym_pogo_stick.envs import PogoStickControlEnv


logger = logging.getLogger(__name__)

class PogoStickDesignEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 100
    }

    def __init__(self,
                 ep_steps=1,
                 sim_steps=400, 
                 reward_function='RewardHeight',
                 specified_height=0.1,  
                 min_spring_k=2038,
                 max_spring_k=7030,
                 min_zeta=0.0,
                 max_zeta=0.01):
       
        
        # Define thresholds for trial limits
        self.rod_max_position = np.inf        # max jump height (m)
        self.rod_min_position = 0             # amount the spring can compress by (m)
        self.rod_max_velocity = np.inf        # max rod velocity (m/s)
        self.act_max_position = 0.25          # length of which actuator can move (m)
        self.act_max_velocity = 2.0           # max velocity of actuator (m/s)

        # variable for logging height reached and spring k chosen
        self.height_reached = 0

        self.state = None
        self.done = False

        # Define env parameters
        self.timestep = 0                 # timestep for number of steps
        self.ep_steps = ep_steps                # maximum number of steps to run
        self.reward_function = reward_function  # variable for choosing what kind of reward function we want to use
        self.specified_height = specified_height
        self.min_spring_k = min_spring_k  
        self.max_spring_k = max_spring_k
        self.min_zeta = min_zeta
        self.max_zeta = max_zeta

        # Define simulation parameters
        self.sim_steps = sim_steps

        # This action space is the range of acceleration mass on the rod
        low_limit = np.array([self.min_spring_k, self.min_zeta])
        high_limit = np.array([self.max_spring_k, self.max_zeta])

        self.action_space = spaces.Box(low=low_limit,
                                       high=high_limit,
                                       dtype=np.float32)
        
        obs_len = int(self.sim_steps)

        low_limit = np.array([self.rod_min_position * np.ones(obs_len),       # max observable jump height
                              -self.rod_max_velocity * np.ones(obs_len),      # max observable jump velocity
                              -self.act_max_position/2 * np.ones(obs_len),    # min observable actuator position
                              -self.act_max_velocity * np.ones(obs_len)])
                                   # max observable actuator velocity
        high_limit = np.array([self.rod_max_position * np.ones(obs_len),      # max observable jump height
                               self.rod_max_velocity * np.ones(obs_len),      # max observable jump velocity
                               self.act_max_position/2 * np.ones(obs_len),    # max observable actuator position
                               self.act_max_velocity * np.ones(obs_len)])     # max observable actuator velocity

        self.observation_space = spaces.Box(low=low_limit,
                                            high=high_limit,
                                            dtype=np.float32)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """ 
        Take one step. Here, we just use a simple Euler integration of the ODE,
        clipping the input and states, if necessary, to meet system limits.
        
        We may need to replace the Euler integration with a more sophisticated
        method, given the nonlinearity of the system.
        """
        self.timestep = self.timestep + 1
        
        self.params = {'k': action[0], 'zeta': action[1]}
        _eval_design(self.params)
        
        # set the return state
        self.height_reached = np.max(self.state[0,:])

        # End the trial when we reach the maximum number of steps
        if self.timestep >= self.ep_steps:
            self.done = True

        # Get the reward depending on the reward function
        reward = self.calc_reward(self.reward_function, self.state)

        return self.state, reward, self.done, {}

    def calc_reward(self, reward_function, timeseries):
        try:
            if reward_function == "RewardHeight": 
                reward = np.max(timeseries[0,:])

            elif reward_function == "RewardSpecifiedHeight":
                height = np.max(timeseries[0,:])
                error = abs(height - self.specified_height) / self.specified_height
                reward = 1 / (error + 1)
                
            else: raise ValueError

        except:
            raise ValueError("REWARD FUNCTION NOT PROPERLY DEFINED PROPERLY")
            print("Proper reward functions are:" ,"\n",
                  "RewardHeight: Rewards jumping high" , "\n",
                  "RewardSpecifiedHeight: Rewards jumping to specified height")
            sys.exit()

        return reward

    def _eval_design(self, params=None):
        # Define the max number of env steps
        ENV_TYPE = "Nonlinear"
        EP_STEPS = 400
        # Set the env
        env = PogoStickControlEnv(numJumps=2,
                                  linear=ENV_TYPE,
                                  epSteps=EP_STEPS, 
                                  evaluating=True,
                                  rewardType=REWARD_FUNCTION,
                                  captureData=False, 
                                  saveDataName=None,
                                  saveDataLocation=None)

        # TODO: Get the model
        model = None
        # Evaluate the agent
        obs = env.reset()
        
        # Set the parameters if need be
        if not params is None:
            env.pogo_stick.modify_design(params=params)

        done, state = False, None
        while not done:
            action, state = model.predict(obs, state=state, deterministic=True)
            obs, reward, done, info = env.step(action)
            # store state is self.state
            for ii in range(len(state)):
                self.state[ii][env.timestep] = state[ii]

        # Remove trailing zeros from self.state
        self.state = self.state[:,:env.timestep+1]
        env.close()

    def reset(self):
        """
        Reset the state of the system at the beginning of the episode.
        We set the rod position and velocity to zero, the actuator velocity to 
        zero, but let the actuator start at random locations along the length
        of the rod
        """ 
        # Reset the timestep, the jumping flag, the landings timestep, the power used list and the height reached list
        self.timestep = 0
        
        self.state = np/zeros((4, self.sim_steps))

        # Reset the done flag
        self.done = False
        
        return np.array(self.state)

    def render(self, mode='human', close=False):
        '''
        Not applicable for this class
        '''
        pass
        