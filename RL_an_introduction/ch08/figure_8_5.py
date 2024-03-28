#######################################################################
# Copyright (C)                                                       #
# 2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)                  #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from maze import *

def figure_8_5():
    # set up a shortcut maze instance
    shortcut_maze = Maze()
    shortcut_maze.START_STATE = [5, 3]
    shortcut_maze.GOAL_STATES = [[0, 8]]
    shortcut_maze.old_obstacles = [[3, i] for i in range(1, 9)]

    # new obstacles will have a shorter path
    shortcut_maze.new_obstacles = [[3, i] for i in range(1, 8)]

    shortcut_maze.max_steps = 6000
    shortcut_maze.obstacle_switch_time = 3000

    dyna_params = DynaParams()
    dyna_params.alpha = 1.0
    dyna_params.planning_steps = 50
    dyna_params.runs = 5
    dyna_params.time_weight = 1e-3
    
    rewards = changing_maze(shortcut_maze, dyna_params)

    # play                
    for i in range(len(dyna_params.methods)):
        plt.plot(rewards[i, :], label=dyna_params.methods[i])
    plt.xlabel('time steps')
    plt.ylabel('cumulative reward')
    plt.legend()
    
    plt.savefig("./plots/figure_8_5.png")
    plt.close()


if __name__ == '__main__':
    figure_8_5()