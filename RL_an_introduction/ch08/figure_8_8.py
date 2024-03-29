#######################################################################
# Copyright (C)                                                       #
# 2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)                  #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

matplotlib.use('Agg')

ACTIONS = [0, 1]
TERMINATION_PROB = 0.1
MAX_STEPS = 20000
EPSILON = 0.1

def argmax(value):
    max_q = np.max(value)
    return np.random.choice([a for a, q in enumerate(value) if q == max_q])


class Task:
    def __init__(self, n_states, b):
        self.n_states = n_states
        self.b = b
        
        self.transition = np.random.randint(n_states, size=(n_states, len(ACTIONS), b))
        self.reward = np.random.randn(n_states, len(ACTIONS), b)
        
    def step(self, state, action):
        if np.random.rand() < TERMINATION_PROB:
            return self.n_states, 0
        next_ = np.random.randint(self.b)
        return self.transition[state, action, next_], self.reward[state, action, next_]
    

def evaluate_pi(q, task):
    runs = 1000
    returns = []
    
    for r in range(runs):
        rewards = 0
        state = 0
        while state < task.n_states:
            action = argmax(q[state])
            state, r = task.step(state, action)
            rewards += r
        returns.append(rewards)
    return np.mean(returns)

def uniform(task, eval_interval):
    performance = []
    q = np.zeros((task.n_states, 2))
    for step in tqdm(range(MAX_STEPS)):
        state = step // len(ACTIONS) % task.n_states
        action = step % len(ACTIONS)
        
        next_states = task.transition[state, action]
        q[state, action] = (1 - TERMINATION_PROB) * np.mean(task.reward[state, action] + np.max(q[next_states, :], axis=1))
        
        if step % eval_interval == 0:
            v_pi = evaluate_pi(q, task)
            performance.append([step, v_pi])
    
    return zip(*performance)

def on_policy(task, eval_interval):
    performance = []
    q = np.zeros((task.n_states, 2))
    state = 0
    for step in tqdm(range(MAX_STEPS)):
        if np.random.rand() < EPSILON:
            action = np.random.choice(ACTIONS)
        else:
            action = argmax(q[state])
        
        next_state, _ = task.step(state, action)
        
        next_states = task.transition[state, action]
        q[state, action] = (1 - TERMINATION_PROB) * np.mean(task.reward[state, action] + np.max(q[next_states, :], axis=1))
        
        if next_state == task.n_states:
            next_state = 0
        state = next_state
        
        if step % eval_interval == 0:
            v_pi = evaluate_pi(q, task)
            performance.append([step, v_pi])
    
    return zip(*performance)
        
def evaluate_task(method, task, max_steps, x_ticks):
    """Function to evaluate a single task with the given method."""
    steps, v = method(task, max_steps / x_ticks)
    return steps, v

def figure_8_8():
    num_states = [1000, 10000]
    branch = [1, 3, 10]
    methods = [on_policy, uniform]

    # average across 30 tasks
    n_tasks = 30

    # number of evaluation points
    x_ticks = 100

    plt.figure(figsize=(10, 20))
    for i, n in enumerate(num_states):
        plt.subplot(2, 1, i+1)
        for b in branch:
            tasks = [Task(n, b) for _ in range(n_tasks)]
            for method in methods:
                steps = None
                value = []

                # Use ProcessPoolExecutor to parallelize task evaluation
                with ProcessPoolExecutor() as executor:
                    # Create a future for each task evaluation
                    futures = [executor.submit(evaluate_task, method, task, MAX_STEPS, x_ticks) for task in tasks]
                    
                    # Wait for the futures to complete and aggregate the results
                    for future in as_completed(futures):
                        steps, v = future.result()
                        value.append(v)

                value = np.mean(np.asarray(value), axis=0)
                plt.plot(steps, value, label=f'b = {b}, {method.__name__}')
        plt.title(f'{n} states')

        plt.ylabel('value of start state')
        plt.legend()

    plt.subplot(2, 1, 2)
    plt.xlabel('computation time, in expected updates')

    plt.savefig('./plots/figure_8_8.png')
    plt.close()


if __name__ == '__main__':
    figure_8_8()