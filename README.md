# Pogo Stick Gym Env

<p align="center">
    <img width="50%" src="pogo_stick_figures\pogo_figure.png" alt="Pogo-Stick Model"/><br>
    <strong>Pogo-Stick Model</strong>
</p>

## System Information
This system is used to research the usefulness of reinforcement learning for defining power conservative control strategies for flexible-legged jumping systems. The details on this system can be found (enter publication link).

## How to Use

Either:

1. See `Concurrent Design/gym_pogo_stick/README.md`
   1. FIXME: Needs updating per concurrent design work
2. See `Learn Control/gym_pogo_stick/README.md`
3. See `Learn Mechanical Parameters/pogo_stick_jumping/README.md`
   1. FIXME: DOES NOT EXIT

### Env Example Parameters

|    Parameter    |                                                            Description                                                            |
|:---------------:|:---------------------------------------------------------------------------------------------------------------------------------:|
|     `numJumps`    |                                 2, `int`: number of jumps the pogo completes to terminate an episode                                |
|      `linear`     |                                 "Linear", `string`: type of spring used in environment. "Nonlinear"                                 |
| `trainRobust`     | False, `bool`: whether or not the env parameters change during training                                                             |
| `epSteps`         | 500, `int`: number of steps to terminate an episode                                                                                 |
| `evaluating`      | False, `bool`: whether or not the position of the actuator is randomly set at reset                                                 |
| `rewardType`      | "Height", `string`: what the agent is going to learn to accomplish. "Efficiency" "SpecHei" "SpHeEf"                                 |
| `specifiedHeight` | 0.05, `float`: the height the agent learns to jump to for rewardType "SpecHei" and "SpHeEf"                                         |
| `captureData`     | False, `bool`: whether or not to capture the time series data for environment [Time, Reward, Input, RodPos, RodVel, ActPos, ActVel] |
| `saveDataName`    | None, `string`: name of the data when captured, default is "Data_xxx_xxx" where xxx_xxx are date and time stamps                    |
| `saveDataLocation`    | None, `string` or `Path`: location to save the data, default is "Captured_Data" within current working directory                      |


See `random_action.py` for example use.