#! /usr/bin/env python

###############################################################################
# poo_stick_jumping_contActionNonlinear.py
#
# Defines a pogo stick jumping environment for use with the openAI Gym.
# This version has a continuous range of inputs for the mass accel. input
#
# Created: 02/03/21
#   - Joshua Vaughan
#   - joshua.vaughan@louisiana.edu
#   - http://www.ucs.louisiana.edu/~jev9637
#
# Modified:
#   * 
#
# TODO:
#   * 02/23/21 - JEV 
#       - [ ] Choose reward function
#       - [ ] Determine proper actuator velocity and acceleration limits
#       - [ ] Decide if we need to use a more sophisticated solver
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
from pogo_stick_jumping.jumping_ODE_Nonlinear import PogoODEnonlinear

logger = logging.getLogger(__name__)

class PogoJumpingEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 100
    }

    def __init__(self,
                 ep_steps=60,
                 sim_step_size=0.01,
                 sim_duration=1, 
                 reward_function='RewardHeight',
                 specified_height=0.1,  
                 min_spring_k=5000,
                 max_spring_k=7200,
                 min_zeta=0.0025,
                 max_zeta=0.0175):
        """
        Initialize with the parameters to match earlier papers from team at 
        Georgia Tech and Dr. Vaughan
        """
        self.gravity = 0.276 * 9.81      # accel. due to gravity (m/s^2)
        self.m_rod = 0.175               # mass of the pogo-stick rod (kg)
        self.m_act = 1.003               # mass of the pogo-stick rod (kg)
        self.mass = self.m_rod + self.m_act  # total mass
        self.f = 11.13                   # natural freq. (rad)
        self.wn = self.f * (2 * np.pi)   # Robot frequency (rad/s)
        self.zeta = 0.01                 # Robot damping ratio
        self.c = 2 * self.zeta * self.wn * self.mass  # Calculate damping coeff

        self.counter = 0                 # counter for number of steps
        self.spacing = 0.75 * (1 / self.f)       # Space commands by one period of oscillation
        
        # Define thesholds for trial limits
        self.rod_max_position = np.inf        # max jump height (m)
        self.rod_min_position = 0             # amount the spring can compress by (m)
        self.rod_max_velocity = np.inf        # max rod velocity (m/s)
        self.act_max_position = 0.25          # length of which actuator can move (m)
        self.act_max_velocity = 2.0           # max velocity of actuator (m/s)
        self.act_max_accel = 63.2             # max acceleration of actuator (m/s^2)
        self.act_distance = 0.008             # distance the actuator can move along the rod
        self.spring_limit = -0.008            # amount the spring can compress by

        # variable for logging height reached and spring k chosen
        self.height_reached = 0

        self.viewer = None
        self.state = None
        self.done = False
        self.x_act_accel = 0.0

        self.ep_steps = ep_steps                # maximum number of steps to run
        self.sim_step_size = sim_step_size           # time between steps in simulation
        self.sim_duration = sim_duration        # how long to simulate for
        self.reward_function = reward_function  # variable for choosing what kind of reward function we want to use
        self.specified_height = specified_height
        self.min_spring_k = min_spring_k  
        self.max_spring_k = max_spring_k
        self.min_zeta = min_zeta
        self.max_zeta = max_zeta

        # This action space is the range of acceleration mass on the rod
        low_limit = np.array([self.min_spring_k, self.min_zeta])
        high_limit = np.array([self.max_spring_k, self.max_zeta])

        self.action_space = spaces.Box(low=low_limit,
                                       high=high_limit,
                                       dtype=np.float32)
        
        obs_len = int(self.sim_duration / self.sim_step_size + 1)

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
        self.counter = self.counter + 1
        self.spring_k = float(action[0])
        self.zeta = float(action[1])

        pogo_stick = PogoODEnonlinear(self.m_act, 
                                      self.m_rod, 
                                      self.spring_k,           # Notice the action is the spring constant
                                      self.zeta, 
                                      self.act_max_accel, 
                                      self.act_max_velocity, 
                                      self.act_distance,
                                      self.spring_limit,
                                      self.spacing)
        
        # TODO: ASA, 09/15/21, Consider using random initial conditions for actuator location
        x_init = 0.0
        x_dot_init = 0.0
        x_act_init = 0.0
        x_act_dot_init = 0.0
        
        x0 = [x_init, x_dot_init, x_act_init, x_act_dot_init]
        
        time, timeseries = pogo_stick.run_simulation(x0, duration=self.sim_duration, max_step=self.sim_step_size)

        # Pull out parts for future use cases
        x = timeseries[0,:]
        x_dot = timeseries[1,:] 
        x_act = timeseries[2,:] 
        x_act_dot = timeseries[3,:]
        
        # set the return state
        self.height_reached = np.max(x)
        self.state = timeseries

        # End the trial when we reach the maximum number of steps
        if self.counter >= self.ep_steps:
            self.done = True
        
        # Log to tensorboard the max height reached and the value of k used

        # record('ep_max_height', max(self.height_reached))
        # record('ep_spring_k', self.spring_k)
        # record('ep_zeta', self.zeta)

        # Get the reward depending on the reward function
        reward = self.calc_reward(self.reward_function, timeseries)

        # Define a boolean on whether we're exceeding limits or not. We can
        # penalize any of these conditions identically in the reward function
        # TODO: 03/16/2021 - Andrew Albright - Not using this at the moment
        # space_limit =  x_act > self.act_max_position \
        #         or x_act < self.act_min_position \
        #         or x_act_dot > self.act_max_velocity \
        #         or x_act_dot < -self.act_max_velocity \

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

    def reset(self):
        """
        Reset the state of the system at the beginning of the episode.
        We set the rod position and velocity to zero, the actuator velocity to 
        zero, but let the actuator start at random locations along the length
        of the rod
        """ 
        # Reset the counter, the jumping flag, the landings counter, the power used list and the height reached list
        self.counter = 0
        
        self.state = np.zeros((4, int(self.sim_duration / self.sim_step_size + 1)))

        # Reset the done flag
        self.done = False
        
        return np.array(self.state)

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        # 600x400 because we're old school, but not 320x240 old.
        screen_width = 600
        screen_height = 400

        scale = 1.0 
        world_height = 3.0
        # scale = screen_width/world_width    # Scale according to width
        scale = screen_height/world_height    # Scale according to height
        
        # Define the pogo diameter and cable width in pixels
        rod_width = 10.0  # pixels
        rod_length= 4 * self.act_max_position * scale
        rod_yOffset = 25  # How far off the bottom of the screen is ground
        
        # Define the trolley size and its offset from the bottom of the screen (pixels)
        actuator_width = 20.0 
        actuator_height = 20.0

        x, x_dot, x_act, x_act_dot = self.state

        if self.viewer is None: # Initial scene setup
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            # define the pogo rod as a polygon, so we can change its length later
            l,r,t,b = -screen_width/2, screen_width, rod_yOffset, -screen_height/2
            self.ground = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.ground.set_color(0.1, 0.1, 0.1)    # very dark gray
            self.viewer.add_geom(self.ground)
            
            # define the pogo rod as a polygon, so we can change its length later
            l,r,t,b = -rod_width/2, rod_width/2, rod_length/2, -rod_width/2
            self.rod = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.rodtrans = rendering.Transform(translation=(screen_width/2, rod_yOffset + x * scale))
            self.rod.add_attr(self.rodtrans)
            self.rod.set_color(0.25, 0.25, 0.25)    # dark gray
            self.viewer.add_geom(self.rod)
            
            # Define the actuator polygon
            l,r,t,b = -actuator_width/2, actuator_width/2, actuator_height/2, -actuator_height/2
            self.actuator = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.actuatortrans = rendering.Transform(translation=(screen_width/2, rod_yOffset + (x + x_act + self.act_max_position)*scale))
            self.actuator.add_attr(self.actuatortrans)
            self.actuator.set_color(0.85, 0.85, 0.85)    # light gray
            self.viewer.add_geom(self.actuator)


        # Move the rod
        self.rodtrans.set_translation(screen_width/2, rod_yOffset + x * scale)
        
        # move the trolley
        self.actuatortrans.set_translation(screen_width/2 , rod_yOffset + (x + x_act + self.act_max_position)*scale)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')
        