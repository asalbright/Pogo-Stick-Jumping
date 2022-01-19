#! /usr/bin/env python

###############################################################################
# __init__.py
#
# initialization for the pogo stick jumping model OpenAI environment
#
# NOTE: Any plotting is set up for output, not viewing on screen.
#       So, it will likely be ugly on screen. The saved PDFs should look
#       better.
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
#   * 
###############################################################################

from gym.envs.registration import register

# Action is the actuator acceleration at every time step
register(
    id='pogo-stick-jumping-v0',
    entry_point='pogo_stick_jumping.pogo_stick_jumping_contAction:PogoJumpingEnv',
)

# Action is the actuator acceleration, the env has a set spring compression limit and the spring is nonlinear
register(
    id='pogo-stick-jumping-v01',
    entry_point='pogo_stick_jumping.pogo_stick_jumping_contActionNonlinear:PogoJumpingEnv',
)

# Action is the spring constant
register(
    id='pogo-stick-jumping-v1',
    entry_point='pogo_stick_jumping.pogo_stick_jumping_springAction:PogoJumpingEnv',
)

# Action is the spring constant, the spring is nonlinear
register(
    id='pogo-stick-jumping-v11',
    entry_point='pogo_stick_jumping.pogo_stick_jumping_springActionNonlinear:PogoJumpingEnv',
)

# Actions are accuator acceleration and a % change spring constant at every time step
register(
    id='pogo-stick-jumping-v2',
    entry_point='pogo_stick_jumping.pogo_stick_jumping_contActionVariableSpring:PogoJumpingEnv',
)
