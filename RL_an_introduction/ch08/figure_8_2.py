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

def figure_8_2():
    dyna_maze = Maze()
    dyna_params = DynaParams()
    
    runs = 10
    episodes = 50
    planning_steps = [0, 5, 50]
    steps = np.zeros((len(planning_steps), episodes))
    
    for run in tqdm(range(runs)):
        for i, planning_step in enumerate(planning_steps):
            dyna_params.planning_steps = planning_step
            q_value = np.zeros(dyna_maze.q_size)
            
            model = TrivalModel()
            for ep in range(episodes):
                steps[i, ep] += dyna_q(q_value, model, dyna_maze, dyna_params)
                
    steps /= runs
    for i in range(len(planning_steps)):
        plt.plot(steps[i, :], label='%d planning steps' % (planning_steps[i]))
    plt.xlabel('episodes')
    plt.ylabel('steps per episode')
    plt.legend()
    
    plt.savefig("./plots/figure_8_2.png")
    plt.close()


if __name__ == '__main__':
    figure_8_2()