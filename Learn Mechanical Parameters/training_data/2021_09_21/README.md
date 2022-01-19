# Description of Training Data
This is a short description of the training data that is within this repository. It is meant to act as a history marker for tracking testing of different training parameters including things like hyperparameter selection.
# Notes: 
Two different training sets were deployed. One to find a design to jump to max height. Another to find a design to jump to a specified height. The one that jumped to a specified height was trying to find a design to jump to 0.01m. which is the standard set in the env. 

## Model Train Date: 09/21/2021

## Training Notes:
### Max Height Agent
```python
# Define the environment ID's and set Environment parameters
env_id = 'pogo-stick-jumping-v11'       # Nonlinear Environment
eval_env_id = 'pogo-stick-jumping-v11'
EPISODE_STEPS = 1
SIM_STEP_SIZE = 0.001
SIM_DURATION = 1
REWARD_FUNCTION = 'RewardHeight'
SPRING_K = 5760
VARIANCE = 1
MIN_SPRING_K = SPRING_K - VARIANCE * SPRING_K
MAX_SPRING_K = SPRING_K + VARIANCE * SPRING_K  
ZETA = 0.01
MIN_ZETA = ZETA - VARIANCE * ZETA
MAX_ZETA = ZETA + VARIANCE * ZETA

# Set up the Training parameters
NUM_TRIALS = 100
TOTAL_SIMS = 1000
ROLLOUT = 100
GAMMA = 0.99

# Set up the training seeds
INITIAL_SEED = 70504
RNG = np.random.default_rng(INITIAL_SEED)
TRIAL_SEEDS = RNG.integers(low=0, high=10000, size=(NUM_TRIALS))
```
### Specific Height Agent
```python
# Define the environment ID's and set Environment parameters
env_id = 'pogo-stick-jumping-v11'       # Nonlinear Environment
eval_env_id = 'pogo-stick-jumping-v11'
EPISODE_STEPS = 1
SIM_STEP_SIZE = 0.001
SIM_DURATION = 1
REWARD_FUNCTION = 'RewardSpecifiedHeight'
SPRING_K = 5760
VARIANCE = 1
MIN_SPRING_K = SPRING_K - VARIANCE * SPRING_K
MAX_SPRING_K = SPRING_K + VARIANCE * SPRING_K  
ZETA = 0.01
MIN_ZETA = ZETA - VARIANCE * ZETA
MAX_ZETA = ZETA + VARIANCE * ZETA

# Set up the Training parameters
NUM_TRIALS = 100
TOTAL_SIMS = 1000
ROLLOUT = 100
GAMMA = 0.99

# Set up the training seeds
INITIAL_SEED = 70504
RNG = np.random.default_rng(INITIAL_SEED)
TRIAL_SEEDS = RNG.integers(low=0, high=10000, size=(NUM_TRIALS))
```