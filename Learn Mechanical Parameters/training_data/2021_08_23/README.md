# Description of Training Data
This is a short description of the training data that is within this repository. It is meant to act as a history marker for tracking testing of different training parameters including things like hyperparameter selection.

## Model Train Date: 08/10/2021

## Environment Notes:
```python
# Define the environment ID's and set Environment parameters
env_id = 'pogo-stick-jumping-v11'       # Nonlinear Environment
eval_env_id = 'pogo-stick-jumping-v11'
EPISODE_SIMS = 5
SIM_STEP_SIZE = 0.001
SIM_DURATION = 1
REWARD_FUNCTION = 'RewardHeight'
spring_k = 5760
variance = 0.75
MIN_SPRING_K = spring_k - variance * spring_k
MAX_SPRING_K = spring_k + variance * spring_k
```

## ENV/TRAINING Parameters: 
```python
# Set up the Training parameters
NUM_TRIALS = 6
TOTAL_SIMS = 100000
ROLLOUT = 1000
GAMMA = 0.99

# Set up the training seeds
INITIAL_SEED = 70504
EVALUATION_FREQ = 10000
RNG = np.random.default_rng(INITIAL_SEED)
TRIAL_SEEDS = RNG.integers(low=0, high=1000, size=(NUM_TRIALS))
```

```python
env = gym.make(env_id)
env.init_variables(NUM_SIMS=EPISODE_SIMS,
                SIM_STEP=SIM_STEP_SIZE,
                SIM_DURATION=SIM_DURATION, 
                REWARD_FUNCTION=REWARD_FUNCTION,  
                MIN_SPRING_K=MIN_SPRING_K,
                MAX_SPRING_K=MAX_SPRING_K,
                SAVE_DATA=False,
                SAVE_NAME=None, 
                SAVE_PATH=False)
```