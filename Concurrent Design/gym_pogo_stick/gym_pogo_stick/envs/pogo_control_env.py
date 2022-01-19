#! /usr/bin/env python


###############################################################################
# pogo_stick_control.py
#
# Defines a pogo stick jumping environment for use with the openAI Gym.
# This version has a continuous range of inputs for the mass accel. input
# Uses pogo_stick_nonlinear.py as pogostick descriptor
#
# Created: 09/15/2021
#   - Andrew Albright
#   - andrew.albright1@louisiana.edu
#
# Notes:
###############################################################################

import os
import sys
import gym
from datetime import datetime
from gym import spaces
from gym.utils import seeding
import logging
import numpy as np
from scipy.integrate import solve_ivp
from pathlib import Path

from gym_pogo_stick.gym_pogo_stick.envs.resources.pogo_stick import PogoStickNonlinear, PogoStickLinear

logger = logging.getLogger(__name__)

class PogoStickControlEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 100
    }

    def __init__(self, 
                 numJumps=2,
                 linear="Linear",
                 trainRobust=False,
                 epSteps=500, 
                 evaluating=False,
                 rewardType="Height",
                 specifiedHeight=0.05,
                 captureData=False, 
                 saveDataName=None,
                 saveDataLocation=None):

        self.num_jumps = numJumps                       # single jump or stutter jump
        self.linear = linear                            # linear or nonlinear spring
        self.robust_training = trainRobust              # train robust agent by varying parameters during training

        self.timestep = 0                               # counter for number of steps
        
        self.ep_steps = epSteps                         # maximum number of steps to run
        self.evaluating = evaluating
        self.reward_type = rewardType
        self.specified_height = specifiedHeight         # height trying to jump to
        self.capture_data = captureData                 # set True to save episode data
        self.data_name = saveDataName                   # path to save data to
        if not saveDataLocation is None:
            self.data_location = Path(saveDataLocation)

        # Reset the environment to get the action and observation space attributes
        self.reset()

        # This action space is the range of acceleration mass on the rod
        low_limit = np.array([-self.pogo_stick.act_max_accel])
        high_limit = np.array([self.pogo_stick.act_max_accel])

        self.action_space = spaces.Box(low=low_limit, high=high_limit)

        # This is the observation space
        high_limit = np.array([self.pogo_stick.rod_max_position,      # max observable jump height
                               self.pogo_stick.rod_max_velocity,      # max observable jump velocity
                               self.pogo_stick.act_max_position,      # max observable actuator position
                               self.pogo_stick.act_max_velocity])     # max observable actuator velocity

        low_limit = np.array([self.pogo_stick.rod_min_position,       # max observable jump height
                              -self.pogo_stick.rod_max_velocity,      # max observable jump velocity
                              self.pogo_stick.act_min_position,       # max observable actuator position
                              -self.pogo_stick.act_max_velocity])     # max observable actuator velocity
        
        self.observation_space = spaces.Box(low=low_limit, high=high_limit)

        self.viewer = None

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def step(self, action):

        # Apply action to the pogo env
        self.pogo_stick.apply_action(action)
        self.timestep += 1

        # Get the observation from the pogo env
        self.state = self.pogo_stick.get_observation()

        if self.timestep >= self.ep_steps or self.pogo_stick.get_num_jumps() >= self.num_jumps:
            self.done = True

        # Calculate the reward based on the observation
        reward = self.calc_reward(self.state, self.reward_type)

        if self.capture_data:
            self._capture_data(reward)

        return self.state, reward, self.done, {}
    
    def calc_reward(self, state, rewardType):
        if rewardType == "Height":
            reward = state[0]

        elif rewardType == "Efficiency":
            height = state[0]
            power_used = self.pogo_stick.get_power_used()
            reward = height / power_used

        elif rewardType == "SpecHei":
            height = state[0]
            if height < self.specified_height:
                reward = -1 * (height - self.specified_height)**2        
            else:
                reward = 2 * (-1 * (height - self.specified_height)**2)

        # TODO: This needs fixing. It does not result in the agent learning the behavior
        #       we are looking for.
        elif rewardType == "SpHeEf":    
            height = state[0]

            if height < self.specified_height:
                rew_error = -1 * (height - self.specified_height)**2        
            else:
                rew_error = 2 * (-1 * (height - self.specified_height)**2)

            rew_power = self.pogo_stick.get_power_used()

            reward = reward * power_used

        else: raise ValueError("\nReward may not be defined properly.\n")

        return reward

    def reset(self):
        """
        Reset the state of the system at the beginning of the episode.
        We set the rod position and velocity to zero, the actuator velocity to 
        zero, but let the actuator start at random locations along the length
        of the rod.
        """ 
        # Set the pogo_stick
        if self.linear == "Linear":
            self.pogo_stick = PogoStickLinear()
        elif self.linear == "Nonlinear":
            self.pogo_stick = PogoStickNonlinear()
        else: raise ValueError("\nEnv type not properly defined. ('Linear' or 'Nonlinear')\n")
        # Reset the counter
        self.timestep = 0
        # Reset the pogo_stick
        self.state = self.pogo_stick.reset_state(self.evaluating, self.robust_training)
        # Reset the done flag
        self.done = False

        if self.capture_data:
            self._create_capture_data_array()

        return self.state

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        # 600x400 because we're old school, but not 320x240 old.
        screen_width = 600
        screen_height = 400

        view_height = 0.75     # Meters
        scale = screen_height / view_height
        
        # Define the pogo parameters in meters
        rod_width = 0.05 * scale
        rod_length= 0.33 * scale
        rod_yOffset = 0.05 * scale  # How far off the bottom of the screen is base of the rod
        
        # Define the actuator size 
        actuator_width = 0.1 * scale 
        actuator_height = 0.1 * scale
        act_yOffset = rod_yOffset + 0.5 * rod_length  # How far up the rod is the actuator

        x, x_dot, x_act, x_act_dot = self.state

        if self.viewer is None: # Initial scene setup
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            # Set the ground plane
            l,r,t,b = -screen_width/2, screen_width, rod_yOffset, -screen_height/2
            self.ground = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.ground.set_color(0.66, 0.16, 0.16)    # very dark gray
            self.viewer.add_geom(self.ground)
            
            # define the pogo rod as a polygon, so we can change its length later
            l,r,t,b = -rod_width/2, rod_width/2, rod_length, 0
            self.rod = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.rodtrans = rendering.Transform(translation=(screen_width/2, rod_yOffset + x * scale))
            self.rod.add_attr(self.rodtrans)
            self.rod.set_color(0.25, 0.25, 0.25)    # dark gray
            self.viewer.add_geom(self.rod)
            
            # Define the actuator polygon
            l,r,t,b = -actuator_width/2, actuator_width/2, actuator_height/2, -actuator_height/2
            self.actuator = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.actuatortrans = rendering.Transform(translation=(screen_width/2, act_yOffset + (x + x_act + self.pogo_stick.act_max_position)*scale))
            self.actuator.add_attr(self.actuatortrans)
            self.actuator.set_color(0.85, 0.85, 0.85)    # light gray
            self.viewer.add_geom(self.actuator)

        if self.state is not None:
            # Move the rod
            rod_x = screen_width/2
            rod_y = rod_yOffset + x * scale
            self.rodtrans.set_translation(rod_x, rod_y)
            
            # move the trolley
            act_x = screen_width/2
            act_y = (rod_y + 0.5*rod_length) + x_act * scale
            self.actuatortrans.set_translation(act_x, act_y)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')
        
    def _create_capture_data_array(self):
        self.ep_data = np.zeros((self.ep_steps + 1, 3 + len(self.state)))         # time, slider height, motors position&velocity
        step_data = [0.0, 0, self.pogo_stick.x_act_accel]  # time, reward, acceleration input
        step_data.extend(self.state)
        self.ep_data[0,:] = step_data
        # If the data name is blank create a temp name to save the data under
        if self.data_name is None: 
            self.temp_data_name = f"Data_{datetime.now().strftime('%m%d%Y_%H%M%S')}"
        # If the date in not appended to the end of the name append it
        elif not datetime.now().strftime('%m%d%Y') in self.data_name:
            self.temp_data_name = self.data_name
            self.data_name = f"{self.temp_data_name}_{datetime.now().strftime('%m%d%Y_%H%M%S')}"
        # If the date is appended to the end, update the time
        else:
            self.data_name = f"{self.temp_data_name}_{datetime.now().strftime('%m%d%Y_%H%M%S')}"
            
        # If the data location is blank, create a temp data location to save to
        if self.data_location is None: 
            data_path = Path.cwd()
            data_path = data_path / "Captured_Data"
            if not os.path.exists(data_path): 
                os.makedirs(data_path)
            self.temp_data_location = data_path
        # If the data path provided is not an existing directory, make it one
        elif not os.path.exists(self.data_location):
            data_path = Path.cwd()
            self.data_location = data_path / self.data_location
            os.makedirs(self.data_location)

    def _capture_data(self, reward):
        # append the data for current step to the episodes data array 
        time = self.pogo_stick.tau * self.timestep # time in seconds
        timestep_data = np.array([time, 
                                 reward,
                                 self.pogo_stick.x_act_accel, 
                                 self.state[0], 
                                 self.state[1], 
                                 self.state[2], 
                                 self.state[3]])
        self.ep_data[self.timestep,:] = timestep_data

        # Save the data
        if self.done:
            # if the data does not fill the array declared for a full length episode
            if self.timestep < self.ep_steps:
                self.ep_data = self.ep_data[0:self.timestep,:]
            # Set the header and save the data
            header = 'Time, Reward, Input, RodPos, RodVel, ActPos, ActVel'
            # Check if we are using the temp file location because one was not provided
            if self.data_name is None: data_name = self.temp_data_name
            else: data_name = self.data_name
            if self.data_location is None: data_location = self.temp_data_location
            else: data_location = self.data_location
            # Set the path and save the file
            data_path = data_location / f"{data_name}.csv"
            np.savetxt(data_path, self.ep_data, header=header, delimiter=',')