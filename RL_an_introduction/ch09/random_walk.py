#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

N_STATES = 1000

STATES = np.arange(1, N_STATES + 1)

START_STATE = 500

END_STATES = [0, N_STATES + 1]

ACTION_LEFT = -1
ACTION_RIGHT = 1
ACTIONS = [ACTION_LEFT, ACTION_RIGHT]

STEP_RANGE = 100

def compute_true_value():
    true_value = np.arange(-1001, 1003, 2) / 1001.0
    
    # while True:
    #     old_vlaue = np.copy(true_value)
    #     for state in STATES:
    #         true_value[state] = 0
    #         for action in ACTIONS:
    #             for step in range(1, STEP_RANGE + 1):
    #                 step *= action
    #                 next_state = state + step
    #                 next_state = max(min(next_state, N_STATES + 1), 0)
    #                 true_value[state] += 1.0 / (2 + STEP_RANGE) * true_value[next_state]
    #     error = np.sum(np.abs(old_vlaue - true_value))
    #     if error < 1e-2:
    #         break
    true_value[0] = true_value[-1] = 0
    
    return true_value

def step(state, action):
    step = np.random.randint(1, STEP_RANGE + 1)
    step *= action
    state += step
    state = max(min(state, N_STATES + 1), 0)
    if state == 0:
        reward = -1
    elif state == N_STATES + 1:
        reward = 1
    else:
        reward = 0
    return state, reward

def get_action():
    if np.random.binomial(1, 0.5) == 1:
        return 1
    return -1

class ValueFunction:
    def __init__(self, num_of_groups):
        self.num_of_groups = num_of_groups
        self.group_size = N_STATES // num_of_groups
        
        self.params = np.zeros(num_of_groups)
        
    def value(self, state):
        if state in END_STATES:
            return 0
        group_index = (state - 1) // self.group_size
        return self.params[group_index]
    
    def update(self, delta, state):
        group_index = (state - 1) // self.group_size
        self.params[group_index] += delta
        
def gradient_monte_carlo(value_function, alpha, distribution=None):
    state = START_STATE
    trajectory = [state]
    
    reward = 0.0
    while state not in END_STATES:
        action = get_action()
        next_state, reward = step(state, action)
        trajectory.append(next_state)
        state = next_state
        
    for state in trajectory[:-1]:
        delta = alpha * (reward - value_function.value(state))
        value_function.update(delta, state)
        if distribution is not None:
            distribution[state] += 1
            
def semi_gradient_temporal_difference(value_function, n, alpha):
    state = START_STATE
    
    states = [state]
    rewards = [0]
    
    time = 0
    
    T = np.inf
    while True:
        time += 1
        
        if time < T:
            action = get_action()
            next_state, reward = step(state, action)
            
            states.append(next_state)
            rewards.append(reward)
            
            if next_state in END_STATES:
                T = time
           
        update_time = time - n     
        if update_time >= 0:
            returns = 0.0
            for t in range(update_time + 1, min(T, update_time + n) + 1):
                returns += rewards[t]
            if update_time + n <= T:
                returns += value_function.value(states[update_time + n])
            state_to_update = states[update_time]
            if not state_to_update in END_STATES:
                delta = alpha * (returns - value_function.value(state_to_update))
                value_function.update(delta, state_to_update)
        if update_time == T - 1:
            break
        state = next_state