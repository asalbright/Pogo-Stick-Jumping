# ********************************************
# Author: Andrew Albright
# Date: 03/31/2021

# File containing useful functions

# ********************************************

import numpy as np
import matplotlib.pyplot as plt
import glob
from pathlib import Path
import pandas
import os
import sys

class RewardHeightNotDefined(Exception):
    pass

def getZipFileName(path=None):
    if path is None:
        print('Path not specified for Zip name finder.')
    
    else:
        files = glob.glob(str(path / '*.zip'))
        print(f'Number of Zip files found: {len(files)}')
        file_names = []
        for f in files:
            file_names.append(os.path.basename(f).split(sep='.')[0])
        print(f'File Names found: {file_names}')

    return file_names

def getReward(reward_function, timeseries):
    try:
        if reward_function == "RewardHeight": 
            # set max height to achieve as 0.25m so that if we reach that height or higher, we give the agent 
            reward = rewardHeight(x=timeseries[0,:], x_dot=timeseries[1,:], x_act=timeseries[2,:], x_act_dot=timeseries[3,:])
        else: raise RewardHeightNotDefined

    except RewardHeightNotDefined:
        print("REWARD FUNCTION NOT PROPERLY DEFINED PROPERLY")
        print()
        sys.exit()

    return reward

def rewardHeight(x, x_dot, x_act, x_act_dot):

    return np.max(x)