import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict

N_STATES = 19

GAMMA = 1

STATES = np.arange(1, N_STATES + 1)

START_STATE = 10

END_STATES = [0, N_STATES + 1]

TRUE_VALUES = np.arange(-20, 22, 2) / 20.0
TRUE_VALUES[0] = TRUE_VALUES[-1] = 0


def n_step_sarsa(n, alpha) -> dict:
    state = START_STATE

    states = [state]
    actions = [np.random.binomial(1, 0.5)]
    rewards = [0]

    Q = np.zeros((N_STATES + 2, 2))

    t = 0

    T = np.inf
    while True:
        t += 1

        if t < T:
            # Take action
            action = actions[t - 1]
            state = states[t - 1]
            if action == 1:
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
            actions.append(action)

            if next_state in END_STATES:
                T = t
            else:
                action = np.random.binomial(1, 0.5)

        tau = t - n
        if tau >= 0:
            G = 0.0

            for i in range(tau + 1, min(T, tau + n) + 1):
                G += np.power(GAMMA, i - tau - 1) * rewards[i]

            if tau + n <= T:
                state_action = (states[tau + n], actions[tau + n])
                G += np.power(GAMMA, n) * Q[state_action[0]][state_action[1]]

            state_action = (states[tau], actions[tau])
            Q[state_action[0]][state_action[1]] += alpha * (
                G - Q[state_action[0]][state_action[1]]
            )

            if tau == T - 1:
                break

    return Q


def figure_7_2():
    steps = np.power(2, np.arange(0, 10))
    alphas = np.arange(0, 1.1, 0.1)
    episodes = 10
    runs = 100

    errors = np.zeros((len(steps), len(alphas)))
    for run in tqdm(range(0, runs)):
        for step_ind, step in enumerate(steps):
            for alpha_ind, alpha in enumerate(alphas):
                for ep in range(0, episodes):
                    Q = n_step_sarsa(step, alpha)
                    errors[step_ind, alpha_ind] += np.sqrt(
                        np.sum(np.power(np.mean(Q, axis=1) - TRUE_VALUES, 2)) / N_STATES
                    )

    errors /= episodes * runs

    for i in range(0, len(steps)):
        plt.plot(alphas, errors[i, :], label="n = %d" % (steps[i]))
    plt.xlabel("alpha")
    plt.ylabel("RMS error")
    # plt.ylim([0.25, 0.55])
    plt.legend()

    # plt.show()
    plt.savefig("./plots/figure_7_2_sarsa.png")
    plt.close()


if __name__ == "__main__":
    figure_7_2()
