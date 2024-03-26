import gymnasium as gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def nstep_on_policy_prediction(V, done, states, rewards):
    if not rewards:
            return 0 if done else V[states[0]]
        
    sub_return = nstep_on_policy_prediction(V, done, states[1:], rewards[1:])
    return rewards[0] + sub_return

def nstep_off_policy_prediction(V, done, states, rewards, isrs):
    if not rewards:
            return 0 if done else V[states[0]]
        
    sub_return = nstep_on_policy_prediction(V, done, states[1:], rewards[1:])
    return isrs[0] * (rewards[0] + sub_return) + (1 - isrs[0]) * V[states[0]]


def td_off_policy_prediction(env, target, behavior, n, num_episodes, alpha=1e-3, simpler=True):
    n_action = env.action_space.n
    n_state = [space.n for space in env.observation_space.spaces]
    
    V = np.zeros(n_state, dtype=float)
    
    history = []
    for episode in range(num_episodes):
        tmp = env.reset()
        state = tmp[0]
        nstep_states = [state]
        nstep_rewards = []
        nstep_isrs = []
        
        done = False
        while nstep_rewards or not done:
            if not done:
                action = np.random.choice(n_action, p=behavior[state])
                state, reward, truncated, terminated, info = env.step(action)
                
                nstep_rewards.append(reward)
                nstep_states.append(state)
                nstep_isrs.append(target[state + (action,)] / behavior[state + (action, )])
                
                if len(nstep_rewards) < n:
                    continue
            
            if simpler is True:
                # undiscounted nstep return (7.1)
                V_target = nstep_on_policy_prediction(V, done, nstep_states, nstep_rewards)
                # multiply by nstep importance sampling ratios
                nstep_isr = np.prod(nstep_isrs)
                # update value function (7.9)
                V[nstep_states[0]] += alpha * nstep_isr * (V_target - V[nstep_states[0]])
            else:
                # undiscounted nstep return (7.13)
                V_target = nstep_off_policy_prediction(V, done, nstep_states, nstep_rewards, nstep_isrs)
                # update value function (7.2)
                V[nstep_states[0]] += alpha * (V_target - V[nstep_states[0]])
                
            del nstep_rewards[0]
            del nstep_states[0]
            del nstep_isrs[0]
            
        history += [np.copy(V)]
    return history


class FixedResetWrapper(gym.ObservationWrapper):
    def reset(self):
        self.env.dealer = [2]
        self.env.player = [1, 2]
        return self.observation(self.env._get_obs())
    
    def observation(self, observation):
         return observation[0], observation[1], int(observation[2])

# env = FixedResetWrapper(gym.make('Blackjack-v1'))
env = gym.make('Blackjack-v1')

V_star = -0.27726
state_star = env.reset()

target = np.zeros([32, 11, 2, 2], dtype=float)
target[:20, :, :, 0] = 1.0
target[20:, :, :, 1] = 1.0

behavior = np.zeros([32, 11, 2, 2], dtype=float)
behavior[:, :, :, 0:2] = 0.5

# env.seed(42)
history0 = td_off_policy_prediction(env, target, behavior, n=2, num_episodes=100_000, alpha=1e-4, simpler=True)

# env.seed(42)
history1 = td_off_policy_prediction(env, target, behavior, n=2, num_episodes=100_000, alpha=1e-4, simpler=False)


matplotlib.reParams['figure.figsize'] = [10, 10]

plt.title("Learning Curve")
plt.xlabel("Episode")
plt.ylabel(f"Value of State {state_star}")
plt.plot([v[state_star] for v in history0], label='(7.1) and (7.9)')
plt.plot([v[state_star] for v in history1], label='(7.2) and (7.13)')
plt.plot([0, 100_000], [V_star, V_star], '-.', color='gray', label='Target')
plt.legend()
plt.showI()