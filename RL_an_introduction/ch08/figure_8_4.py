import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from maze import *

def figure_8_4():
    # set up a blocking maze instance
    blocking_maze = Maze()
    blocking_maze.START_STATE = [5, 3]
    blocking_maze.GOAL_STATES = [[0, 8]]
    blocking_maze.old_obstacles = [[3, i] for i in range(0, 8)]

    # new obstalces will block the optimal path
    blocking_maze.new_obstacles = [[3, i] for i in range(1, 9)]

    blocking_maze.max_steps = 3000
    blocking_maze.obstacle_switch_time = 1000

    dyna_params = DynaParams()
    dyna_params.alpha = 1.0
    dyna_params.planning_steps = 10
    dyna_params.runs = 20
    
    dyna_params.time_weight = 1e-4
    
    rewards = changing_maze(blocking_maze, dyna_params)

    # play                
    for i in range(len(dyna_params.methods)):
        plt.plot(rewards[i, :], label=dyna_params.methods[i])
    plt.xlabel('time steps')
    plt.ylabel('cumulative reward')
    plt.legend()
    
    plt.savefig("./plots/figure_8_4.png")
    plt.close()


if __name__ == '__main__':
    figure_8_4()