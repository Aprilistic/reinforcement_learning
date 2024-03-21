import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tqdm import tqdm

N_STATES = 19

GAMMA = 1

STATES = np.arange(1, N_STATES + 1)

START_STATE = 10

END_STATES = [0, N_STATES + 1]

TRUE_VALUES = np.arange(-20, 22, 2) / 20.0
TRUE_VALUES[0] = TRUE_VALUES[-1] = 0

def temporal_difference(value, n, alpha):
    state = START_STATE
    
    states = [state]
    rewards = [0]
    
    time = 0
    
    T = np.inf
    while True:
        time += 1
        
        if time < T:
            if np.random.binomial(1, 0.5) == 1:
                next_state = state + 1
            else:
                next_state = state - 1
            
            if next_state == 0:
                reward = -1
            elif next_state == 20:
                reward = 1
            else:
                reward = 0
                
            states.append(next_state)
            rewards.append(reward)
            
            if next_state in END_STATES:
                T = time
        
        update_time = time - n   
        if update_time >= 0:
            returns = 0.0
            
            for t in range(update_time + 1, min(T, update_time + n) + 1):
                returns += pow(GAMMA, t - update_time - 1) * rewards[t]
            
            if update_time + n <= T:
                returns += pow(GAMMA, n) * value[states[update_time + n]]
            state_to_update = states[update_time]
            
            if not state_to_update in END_STATES:
                value[state_to_update] += alpha * (returns - value[state_to_update])
            
            if update_time == T - 1:
                break
            state = next_state
            
def figure_7_2():
    steps = np.power(2, np.arange(0, 10))
    alphas = np.arange(0, 1.1, 0.1)
    episodes = 10
    runs = 100
    
    errors = np.zeros((len(steps), len(alphas)))
    for run in tqdm(range(0, runs)):
        for step_ind, step in enumerate(steps):
            for alpha_ind, alpha in enumerate(alphas):
                value = np.zeros(N_STATES + 2)
                for ep in range(0, episodes):
                    temporal_difference(value, step, alpha)
                    errors[step_ind, alpha_ind] += np.sqrt(np.sum(np.power(value - TRUE_VALUES, 2)) / N_STATES)
    
    errors /= episodes * runs
    
    for i in range(0, len(steps)):
        plt.plot(alphas, errors[i, :], label='n = %d' % (steps[i]))
    plt.xlabel('alpha')
    plt.ylabel('RMS error')
    plt.ylim([0.25, 0.55])
    plt.legend()
    
    plt.show()
    plt.savefig('./plots/figure_7_2.png')
    plt.close()

if __name__ == '__main__':
    figure_7_2()