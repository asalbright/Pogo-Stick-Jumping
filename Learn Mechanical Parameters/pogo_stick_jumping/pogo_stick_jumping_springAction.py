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
from functions import getReward
from pogo_stick_jumping.jumping_ODE import PogoODE


logger = logging.getLogger(__name__)


class PogoJumpingEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 100
    }

    def __init__(self, 
                 EVALUATING=False, 
                 NUM_SIMS=60,
                 SIM_STEP=0.01,
                 SIM_DURATION=1, 
                 REWARD_FUNCTION='RewardHeight',  
                 MIN_SPRING_K=200000,
                 MAX_SPRING_K=200000,
                 SAVE_DATA=False,
                 SAVE_NAME=None, 
                 SAVE_PATH=False):
        """
        Initialize with the parameters to match earlier papers from team at 
        Georgia Tech and Dr. Vaughan
        """
        self.gravity = 9.81              # accel. due to gravity (m/s^2)
        self.m_rod = 0.175               # mass of the pogo-stick rod (kg)
        self.m_act = 1.003               # mass of the pogo-stick rod (kg)
        self.mass = self.m_rod + self.m_act  # total mass
        self.f = 11.13                   # natural freq. (rad)
        self.wn = self.f * (2 * np.pi)   # Robot frequency (rad/s)
        self.zeta = 0.01                 # Robot damping ratio
        self.c = 2 * self.zeta * self.wn * self.mass  # Calculate damping coeff

        self.counter = 0                 # counter for number of steps
        self.Spacing = 0.5 * (1 / self.f)       # Space commands by one period of oscillation

        # variable for adjusting the length and step size of the simulation
        self.NUM_SIMS = NUM_SIMS  # maximum number of steps to run
        self.sim_max_steps = SIM_STEP           # time between steps in simulation
        self.sim_duration = SIM_DURATION        # how long to simulate for
        
        self.REWARD_FUNCTION = REWARD_FUNCTION  # variable for choosing what kind of reward function we want to use
        
        # Define spring contaant range
        self.min_springConstant = MIN_SPRING_K
        self.max_springConstant = MAX_SPRING_K
        
        self.SAVE_DATA = SAVE_DATA          # set True to save episode data
        self.SAVE_PATH = SAVE_PATH       # path to save data to
        self.SAVE_NAME = SAVE_NAME
        self.EVALUATING = EVALUATING     # Flag if using the env for evaluating or training

        # Define thesholds for trial limits
        self.rod_max_position = np.inf        # max jump height (m)
        self.rod_min_position = 0             # amount the spring can compress by (m)
        self.rod_max_velocity = np.inf        # max rod velocity (m/s)
        self.act_max_position = 0.25          # length of which actuator can move (m)
        self.act_max_velocity = 2.0           # max velocity of actuator (m/s)
        self.act_max_accel = 63.2             # max acceleration of actuator (m/s^2)

        # variable for logging height reached and spring k chosen
        self.height_reached = []
        self.spring_k_chosen = []

        self.seed()
        self.viewer = None
        self.state = None
        self.done = False
        self.x_act_accel = 0.0

    def init_variables(self, 
                EVALUATING=False, 
                NUM_SIMS=60,
                SIM_STEP=0.01,
                SIM_DURATION=1, 
                REWARD_FUNCTION='RewardHeight',  
                MIN_SPRING_K=50000,
                MAX_SPRING_K=350000,
                SAVE_DATA=False,
                SAVE_NAME=None, 
                SAVE_PATH=False):

        self.EVALUATING = EVALUATING            # Flag if using the env for evaluating or training
        self.NUM_SIMS = NUM_SIMS                # maximum number of steps to run
        self.sim_max_steps = SIM_STEP           # time between steps in simulation
        self.sim_duration = SIM_DURATION        # how long to simulate for
        self.REWARD_FUNCTION = REWARD_FUNCTION  # variable for choosing what kind of reward function we want to use
        self.min_springConstant = MIN_SPRING_K  
        self.max_springConstant = MAX_SPRING_K
        self.SAVE_DATA = SAVE_DATA              # set True to save episode data
        self.SAVE_NAME = SAVE_NAME
        self.SAVE_PATH = SAVE_PATH              # path to save data to
        # make the path for saving the data
        if self.SAVE_DATA:
            if not self.SAVE_PATH:
                save_path = Path.cwd()
                self.SAVE_PATH = save_path.joinpath('logs')
                if not os.path.exists(self.SAVE_PATH):
                    os.makedirs(self.SAVE_PATH)
            else:
                if not os.path.exists(self.SAVE_PATH):
                    os.makedirs(self.SAVE_PATH)

                # This action space is the range of acceleration mass on the rod
        self.action_space = spaces.Box(low=self.min_springConstant,
                                       high=self.max_springConstant,
                                       shape=(1,))
        
        obs_len = int(self.sim_duration / self.sim_max_steps + 1)

        high_limit = np.array([self.rod_max_position * np.ones(obs_len),      # max observable jump height
                               self.rod_max_velocity * np.ones(obs_len),      # max observable jump velocity
                               self.act_max_position/2 * np.ones(obs_len),    # max observable actuator position
                               self.act_max_velocity * np.ones(obs_len)])     # max observable actuator velocity

        low_limit = np.array([self.rod_min_position * np.ones(obs_len),       # max observable jump height
                              -self.rod_max_velocity * np.ones(obs_len),      # max observable jump velocity
                              -self.act_max_position/2 * np.ones(obs_len),    # min observable actuator position
                              -self.act_max_velocity * np.ones(obs_len)])     # max observable actuator velocity
                              
        self.observation_space = spaces.Box(high_limit, low_limit)

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
        # Use random initial conditions for actuator location
        x0 = np.array([0,  # 0 initial height above ground
                       0,  # 0 initial vertical velocity
                       0,  # 0 initial actuator position (midpoint on pogo)
                       0]) # 0 initial actuator velocity
        
        pogo_stick = PogoODE(self.m_act, 
                             self.m_rod, 
                             self.spring_k,           # Notice the action is the spring constant
                             self.zeta, 
                             self.act_max_accel, 
                             self.act_max_velocity, 
                             self.act_max_position/3, # How far to move actuator 
                             self.Spacing)            # time spacing between bang-bang commands
        
        time, timeseries = pogo_stick.run_simulation(x0, duration=self.sim_duration, max_step=self.sim_max_steps)

        x = timeseries[0,:]
        x_dot = timeseries[1,:] 
        x_act = timeseries[2,:] 
        x_act_dot = timeseries[3,:]
        
        # set the return state
        self.state = timeseries

        # End the trial when we reach the maximum number of steps
        if self.counter >= self.NUM_SIMS:
            self.done = True
        
        # Append the max height reached to the list for logging in tensorboard
        self.height_reached.append(np.max(x))
        self.spring_k_chosen.append(self.spring_k)
        print(f'height reached: MAX: {np.max(x)}, MIN: {np.min(x)}')
        print(f'spring k usedK {self.spring_k}')

        # Get the reward depending on the reward function
        reward = getReward(self.REWARD_FUNCTION, timeseries)

        # Define a boolean on whether we're exceeding limits or not. We can
        # penalize any of these conditions identically in the reward function
        # TODO: 03/16/2021 - Andrew Albright - Not using this at the moment
        # space_limit =  x_act > self.act_max_position \
        #         or x_act < self.act_min_position \
        #         or x_act_dot > self.act_max_velocity \
        #         or x_act_dot < -self.act_max_velocity \

        # TODO: 04/28/2021 - ASA - Want to consider how we are saving the data
        # append the data for current step to the episodes data array 
        sim_data = self.state
        self.episode_data.append(sim_data)

        # If the episode is finished, we create the csv of the episode data, including a text header.
        if self.done and self.SAVE_DATA:
            header = []
            for ii in range(len(self.episode_data)):
                header.append(f"SIM{ii}")
            data_filename = f'EpisodeData_{self.SAVE_NAME}.csv'
            data_path = self.SAVE_PATH / data_filename
            np.savetxt(data_path, self.episode_data, header=header, delimiter=',')

        return self.state, reward, self.done, {}

    def reset(self):
        """
        Reset the state of the system at the beginning of the episode.
        We set the rod position and velocity to zero, the actuator velocity to 
        zero, but let the actuator start at random locations along the length
        of the rod
        """ 
        # publish max height reached and spring k used to get to that height to tensorboard for tracking during training
        if len(self.height_reached):
            record('ep_max_height', max(self.height_reached))
            record('ep_spring_k', self.spring_k_chosen[self.spring_k_chosen.index(max(self.height_reached))])

        # Reset the counter, the jumping flag, the landings counter, the power used list and the height reached list
        self.counter = 0
        self.spring_k_chosen = []
        self.height_reached = []
        
        self.state = np.zeros((4, int(self.sim_duration / self.sim_max_steps + 1)))

        # Reset the done flag
        self.done = False
        
        # If we are saving data, set up the array to save the data into until
        # we save it at the end of the episode
        self.episode_data = []

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
        