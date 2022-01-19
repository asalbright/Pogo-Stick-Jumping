###############################################################################
# __init__.py
#
# initialization for the pogo stick jumping model OpenAI environment
#
#
# Created: 09/16/2021
#   - Andrew Albright
#   - andrew.albright1@louisiana.edu
#
# Notes:
#    
###############################################################################

from gym.envs.registration import register

# Action is the actuator acceleration at every time step
register(
    id='pogo-stick-control-v0',
    entry_point='gym_pogo_stick.envs:PogoStickControlEnv',
)