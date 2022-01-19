#! /usr/bin/env python

###############################################################################
# custom_callbacks.py
#
# Callbacks to use during training
#
# Created: 01/04/2022
#   - Andrew Albright
#   - andrew.albright1@louisiana.edu
#
# Notes:
###############################################################################

import os
import numpy as np
from typing import List, Set, Dict, Tuple, Optional
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import TD3

from gym_pogo_stick.gym_pogo_stick.envs.pogo_design_env import PogoStickDesignEnv

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        
        self.heights = []
        self.eposide = -1

    def _on_step(self) -> bool:
        # Capture the height at the current timestep
        env = self.training_env.envs[0].env
        state = env.state
        height = env.state[0]

        if env.timestep == 1:
            self.heights.append([])
            self.eposide += 1
        
        self.heights[self.eposide].append(height)
        
        return True
        
    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        max_height = np.max(self.heights)
        self.logger.record('max_height', max_height)

        self.heights = []
        self.eposide = -1

        pass


class TrainingDesignCallback(BaseCallback):
    """
    Custom callback for updating the environment design.
    """

    def __init__(self, verbose=0):
        super(TrainingDesignCallback, self).__init__(verbose)

        self.episode_steps = 1
        self.sim_step_size = 0.001
        self.sim_steps = 400
        self.reward_function = 'RewardHeight'

        # set up the training parameters
        self.num_trials = 100
        self.total_sims = 1000
        self.rollout = 100
        self.gamma = 0.99

    def _on_step(self) -> bool:        
        return True
        
    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        # Get the new env design parameters from the design env
        design_params = self._train_design()

        # Push the design parameters to the control env
        _update_design_params(design_params)

    def _train_design(self) -> Dict:
        # Set up training env
        env = PogoStickDesignEnv(ep_steps=self.episode_steps,
                                 sim_steps=self.sim_steps, 
                                 reward_function=self.reward_function)

        # set the trial seed for use during training
        trial_seed = int(self.model.seed)

        env.seed(seed=self.model.seed)

        # wrap the env in modified monitor which plots to tensorboard the jumpheight
        env = Monitor(env)

        # create the model
        # open tensorboard with the following bash command: tensorboard --logdir ./logs/
        buffer_size = self.total_sims + 1
        model = TD3("MlpPolicy", 
                    env, 
                    verbose=1, 
                    buffer_size=buffer_size, 
                    learning_starts=self.rollout, 
                    seed=self.model.seed, 
                    gamma=self.gamma)

        # train the agent
        # TODO: 01/04/2022 - Create a callback or a variable that is updated with the design parameters
        # callback = LogMechParamsCallback()

        model.learn(total_timesteps=self.total_sims, callback=callback, tb_log_name=f'{data_name}_{int(seed)}_log')

        # TODO: 01/04/2022 - Return the design parameters
        return None # design_params

    def _update_design_params(self) -> None:
        # TODO: 01/04/2022 - Update the design parameters in the control env
        pass