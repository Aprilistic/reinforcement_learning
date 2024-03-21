from typing import Any
import numpy as np
import matplotlib.pyplot as plt

from envs.random_walk_env import RandomWalk


def random_policy() -> int:
    return np.random.choice(2)


def TD_error_predictor(
    alpha: float = 0.1, gamma: float = 1.0, num_episodes=100, n=1
) -> np.ndarray:
    env = RandomWalk()
    total_states = len(env.state_space) + 2
    V = np.zeros(shape=(total_states), dtype=float)
    V[1:-1] += 0.5  # Initialize non-terminal states with a value of 0.5

    V_history = np.zeros(shape=(num_episodes, total_states))

    for episode in range(num_episodes):
        env.reset()
        state = env.initial_idx
        terminated = False
        trajectory = [(state, 0)]  # Initial state with 0 reward
        T = float("inf")
        t = 0
        TD_errors_sum = 0  # Initialize sum of TD errors

        while not terminated:
            if t < T:
                action = random_policy()
                next_state, reward, terminated = env.step(action)
                trajectory.append((next_state, reward))

                if not terminated:
                    TD_error = reward + gamma * V[next_state] - V[state]
                    TD_errors_sum += TD_error  # Accumulate TD errors

                    if t % n == 0:
                        V[state] += alpha * TD_errors_sum
                        TD_errors_sum = 0  # Reset the sum of TD errors

                    state = next_state
                else:
                    T = t + 1

            t += 1

        # Apply the accumulated TD errors at the end of the episode
        for s, _ in trajectory:
            V[s] += (
                alpha * TD_errors_sum / len(trajectory)
            )  # Apply evenly or based on specific logic

        V_history[episode] = np.copy(V)
    
    return V_history


def TD_n_predictor(
    alpha: float = 0.1, gamma: float = 1.0, num_episodes=100, n=1
) -> np.ndarray:
    env = RandomWalk()
    total_states = len(env.state_space) + 2
    V = np.zeros(shape=(total_states), dtype=float)
    V[1:-1] += 0.5  # Initialize non-terminal states with a value of 0.5

    V_history = np.zeros(shape=(num_episodes, total_states))

    for episode in range(num_episodes):
        env.reset()
        state = env.initial_idx
        terminated = False
        trajectory = [(state, 0)]  # Initial state with 0 reward
        T = float("inf")
        t = 0

        while True:
            if t < T:
                action = random_policy()
                next_state, reward, terminated = env.step(action)
                trajectory.append((next_state, reward))

                if terminated:
                    T = t + 1
                else:
                    state = next_state

            tau = t - n + 1  # Time step being updated
            if tau >= 0:
                G = 0  # Initialize G for sum of discounted rewards
                # Calculate the sum of discounted rewards (G)
                for i in range(tau + 1, min(tau + n, T) + 1):
                    G += gamma ** (i - tau - 1) * trajectory[i][1]

                # Add the value of the state n steps ahead, if it exists
                if tau + n < T:
                    G += gamma**n * V[trajectory[tau + n][0]]

                # Calculate TD error
                TD_error = G - V[trajectory[tau][0]]

                # Update the value function using the TD error
                V[trajectory[tau][0]] += alpha * TD_error

            if tau == T - 1:
                break  # Break if the last state to be updated is reached

            t += 1
        
        V_history[episode] = np.copy(V)

    return V_history


def estimate_value(num_episodes: int, true_value: np.ndarray) -> None:
    V_hist_single = TD_error_predictor(alpha=0.1, num_episodes=num_episodes)
    sel_epochs = (0, 9, 99)
    selected_hist = V_hist_single[sel_epochs, 1:-1]

    font_dict = {"fontsize": 11}
    x_axis = np.arange(5, dtype=int)
    colors = ["mediumseagreen", "steelblue", "orchid"]

    plt.figure(figsize=(6, 6), dpi=150)
    plt.grid(c="lightgray")
    plt.margins(0.02)
    plt.xticks(x_axis, ["A", "B", "C", "D", "E"])
    for i, spine in enumerate(plt.gca().spines.values()):
        if i in [0, 2]:
            spine.set_linewidth(1.5)
            continue
        spine.set_visible(False)
    plt.xlabel("State", fontdict=font_dict)
    plt.ylabel("Estimated value", fontdict=font_dict)

    plt.scatter(x_axis, true_value, c="black", s=10.0)
    plt.plot(x_axis, true_value, c="black", linewidth=1.6, label="True values")

    for i, hist in enumerate(selected_hist):
        plt.scatter(x_axis, hist, s=7.0, c=colors[i])
        plt.plot(
            x_axis, hist, linewidth=1.2, c=colors[i], label=f"sweep {sel_epochs[i]+1}"
        )
    plt.title("The values learned after $n$ sweeps", fontweight="bold", fontsize=13)
    plt.legend(loc=4)
    plt.savefig("./plots/example_7_2/td_error.png")
    plt.show()

if __name__ == "__main__":
    # The true value of the random policy for RW is provided as follows
    true_value = np.arange(0, 1.1, 1 / 6)[1:-1]
    num_runs = 100
    num_episodes = 150

    estimate_value(num_episodes=num_episodes, true_value=true_value)
