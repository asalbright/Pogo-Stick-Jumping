###############################################################################
# pogo_stick_nonlinear.py
#
# class describing the pogo-stick used in for CRAWLAB research
#
#
# Created: 09/16/2021
#   - Andrew Albright
#   - andrew.albright1@louisiana.edu
#
# Notes: The parameters for this system are from work completed at Georgia Tech
#    
###############################################################################
import os
import math
import numpy as np
import sys
class PogoStick:
    def __init__(self):
        """
        Initialize with the parameters to match earlier papers from team at 
        Georgia Tech and Dr. Vaughan
        """
        # gravity would be multiplied by 0.276 according to the paper
        self.gravity = 9.81              # accel. due to gravity (m/s^2)
        self.m_rod = 0.175               # mass of the pogo-stick rod (kg)
        self.m_act = 1.003               # mass of the pogo-stick rod (kg)
        self.mass = self.m_rod + self.m_act  # total mass

        self.k = 5760
        self.wn = np.sqrt(self.k/self.mass)   # ~= 70
        self.wn_nom = self.wn            # used for changing the spring constant

        self.zeta = 0.01                 # Robot damping ratio
        self.zeta_nom = self.zeta        # used for changing the damping ratio
        self.c = 2 * self.zeta * self.wn * self.mass  # ~= 1.65
        
        self.tau = 0.01                  # seconds between state updates
        self.spring_limit = -0.008       # spring compression limit


        self.jumping = False             # Flag for defining if the system is jumping
        self.landed = 0                  # Counter for number of times the system has landed
        self.power_used = []
        self.height_reached = []

        self.rod_max_position = np.inf   # max jump height (m)
        self.rod_min_position = -0.125   # amount the spring can compress by (m)
        self.rod_max_velocity = np.inf   # max rod velocity (m/s)
        self.act_max_position = 0.008    # length of which actuator can move (m)
        self.act_min_position = 0.0      # lower limit of the actuator (m)
        self.act_max_velocity = 1.0      # max velocity of actuator (m/s)
        self.act_max_accel = 10.0        # max acceleration of actuator (m/s^2)

        self.x_act_accel = 0.0
        self.state = None
    
    def apply_action(self, action):
        self.x, self.x_dot, self.x_act, self.x_act_dot = self.state
        
        # Get the action and clip it to the min/max trolley accel
        self.x_act_accel = np.clip(action[0], -self.act_max_accel, self.act_max_accel)
        
        # If the actuator is at a limit, don't let it accelerate in that direction
        if self.x_act >= self.act_max_position and self.x_act_accel > 0:
            self.x_act_accel = 0
        elif self.x_act <= self.act_min_position and self.x_act_accel < 0:
            self.x_act_accel = 0
        
        # Update the actuator states
        self.x_act_dot = self.x_act_dot + self.tau * self.x_act_accel

        # Keep velocity within limits
        self.x_act_dot = np.clip(self.x_act_dot, -self.act_max_velocity, self.act_max_velocity)
        
        self.x_act = self.x_act + self.tau * self.x_act_dot
        
        # Keep actuator position within limits
        self.x_act = np.clip(self.x_act, self.act_min_position, self.act_max_position)
        
        #determine if the system is in the air
        if self.x > 0:
            self.contact = 0
        else:
            self.contact = 1
        
        # Update the rod states in the child classes

    def limit_spring_compression(self):
        # If the spring is at its compression limit, stop the pogo from compressing futher
        if self.x <= self.spring_limit:
            x = self.spring_limit
            if self.x_dot < 0:
                self.x_dot = 0
            if self.x_ddot < 0:
                self.x_ddot = 0

        # Set the return state
        self.state = (self.x, self.x_dot, self.x_act, self.x_act_dot)

        # If the pogo is jumping, add to the number of jumps
        if self.x > 0 and self.jumping is False:
                self.jumping = True
        elif self.x <=0 and self.jumping is True:
                self.landed = self.landed + 1
                self.jumping = False

        # calculate the power used by the actuator
        power = float(abs(self.m_act * self.x_act_accel * self.x_act_dot))
        self.power_used.append(power)
        self.height_reached.append(float(self.x))

    def get_power_used(self):
        return np.sum(self.power_used)

    def get_num_jumps(self):
        return self.landed

    def get_observation(self):
        return np.array(self.state)

    def modify_design(self, percent_change=0.30, params=None):
        # If the params are specified, use them
        if not params is None:
            self.k = params["k"]
            self.zeta = params["zeta"]

        # If the params are not specified, use random values
        else:
            # Pick a nominal spring k such that w_n varies +/- 30% of nominal value
            # k = 2038 to 7030
            # Get max k and min k
            wn_min = self.wn_nom - percent_change*self.wn_nom
            wn_max = self.wn_nom + percent_change*self.wn_nom
            k_min = wn_min**2 * self.mass
            k_max = wn_max**2 * self.mass

            # Pick a damping ratio range from 0 to 0.01
            # Get max zeta and min zeta
            zeta_min = 0.0
            zeta_max = self.zeta_nom
            
            self.zeta = np.random.uniform(low=zeta_min, high=zeta_max)
            self.k = np.random.uniform(low=k_min, high=k_max)
    
        # Update natural frequency and damping constant
        self.wn = np.sqrt(self.k/self.mass)
        self.c = 2 * self.zeta * self.wn * self.mass

    def reset_state(self, evaluating=False, vary_parameters=False):
        if evaluating: 
            self.state = np.array([self.rod_zero,               # compressed sprint height above ground
                                   0,                           # 0 initial vertical velocity
                                   (self.act_max_position)/2,   # center position
                                   0])                          # 0 initial actuator velocity
        # during training randomly place the actuator along its path
        else:
            act_pos = np.random.uniform(low=self.act_min_position, high=self.act_max_position)
            self.state = np.array([self.rod_zero,               # compressed spring height above ground
                                   0,                           # 0 initial vertical velocity
                                   act_pos,                     # Random point along stroke
                                   0])                          # 0 initial actuator velocity
        # if we are training a robust agent, change the model slightly
        if vary_parameters:
            self.modify_design()

        self.jumping = False
        self.landed = 0
        self.power_used = []
        self.height_reached = []

        return self.state


class PogoStickLinear(PogoStick):
    def __init__(self):
        super().__init__()
        self.rod_zero = -(self.mass * self.gravity / self.k) 
    
    def apply_action(self, action):
        # Use the apply action in the Parent class
        super().apply_action(action)

        # Update the rod state, only allowing the spring and damper to act if
        # the rod is in contact with the ground.        
        self.x_ddot = -self.contact / self.mass * (self.k * self.x + self.c * self.x_dot) - self.m_act/self.mass * self.x_act_accel - self.gravity
        self.x_dot = self.x_dot + self.tau * self.x_ddot
        self.x = self.x + self.tau * self.x_dot

        super().limit_spring_compression()

        return np.array(self.state)


class PogoStickNonlinear(PogoStick):
    def __init__(self):
        super().__init__()
        #FIXME: This is according to the linear case, but it is not correct
        self.rod_zero = -(self.mass * self.gravity / self.k)

    def apply_action(self, action):
        # Use the apply action in the Parent class
        super().apply_action(action)

        # Update the rod state, only allowing the spring and damper to act if the rod is in contact with the ground        
        self.x_ddot = -self.contact / self.mass * (self.k * self.x + self.c * self.x_dot) - self.m_act/self.mass * self.x_act_accel - self.gravity
        self.x_dot = self.x_dot + self.tau * self.x_ddot
        self.x = self.x + self.tau * self.x_dot

        super().limit_spring_compression()

        return self.state