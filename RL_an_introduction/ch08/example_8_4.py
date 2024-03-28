#######################################################################
# Copyright (C)                                                       #
# 2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)                  #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from maze import *

def run_experiment(run, method_names, models, methods, params, mazes, num_of_mazes, params_dyna):
    backups = np.zeros((2, num_of_mazes))
    for i in range(len(method_names)):
        for mazeIndex, maze in enumerate(mazes):
            print(f'run {run}, {method_names[i]}, maze size {maze.WORLD_HEIGHT * maze.WORLD_WIDTH}')
            
            q_value = np.zeros(maze.q_size)
            steps = []
            model = models[i]()
            
            while True:
                steps.append(methods[i](q_value, model, maze, params[i]))
                
                if check_path(q_value, maze):
                    break
            
            backups[i, mazeIndex] = np.sum(steps)
    
    # Dyna-Q performs several backups per step
    backups[1, :] *= params_dyna.planning_steps + 1
    return backups


def example_8_4():
    original_maze = Maze()
    
    params_prioritized = DynaParams()
    params_prioritized.theta = 0.001
    params_prioritized.planning_steps = 5
    params_prioritized.alpha = 0.5
    params_prioritized.gamma = 0.95
    
    params_dyna = DynaParams()
    params_dyna.planning_steps = 5
    params_dyna.alpha = 0.5
    params_dyna.gamma = 0.95
    
    params = [params_prioritized, params_dyna]
    models = [PriorityModel, TrivialModel]
    method_names = ['Prioritized Sweeping', 'Dyna-Q']
    
    num_of_mazes = 3
    
    mazes = [original_maze.extend_maze(i) for i in range(1, num_of_mazes + 1)]
    methods = [prioritized_sweeping, dyna_q]
    
    runs = 3
    
    backups = np.zeros((runs, 2, num_of_mazes))
    
    for run in range(0, runs):
        for i in range(0, len(method_names)):
            for mazeIndex, maze in zip(range(0, len(mazes)), mazes):
                print('run %d, %s, maze size %d' % (run, method_names[i], maze.WORLD_HEIGHT * maze.WORLD_WIDTH))
                
                q_value = np.zeros(maze.q_size)
                steps = []
                model = models[i]()
                
                while True:
                    steps.append(methods[i](q_value, model, maze, params[i]))
                    
                    if check_path(q_value, maze):
                        break
                
                backups[run, i, mazeIndex] = np.sum(steps)
                
    backups = backups.mean(axis=0)

    # Dyna-Q performs several backups per step
    backups[1, :] *= params_dyna.planning_steps + 1

    for i in range(0, len(method_names)):
        plt.plot(np.arange(1, num_of_mazes + 1), backups[i, :], label=method_names[i])
    plt.xlabel('maze resolution factor')
    plt.ylabel('backups until optimal solution')
    plt.yscale('log')
    plt.legend()

    plt.savefig('./plots/example_8_4.png')
    plt.close()             

def example_8_4_parallel():
    original_maze = Maze()
    
    params_prioritized = DynaParams()
    params_prioritized.theta = 0.001
    params_prioritized.planning_steps = 5
    params_prioritized.alpha = 0.5
    params_prioritized.gamma = 0.95
    
    params_dyna = DynaParams()
    params_dyna.planning_steps = 5
    params_dyna.alpha = 0.5
    params_dyna.gamma = 0.95
    
    params = [params_prioritized, params_dyna]
    models = [PriorityModel, TrivialModel]
    method_names = ['Prioritized Sweeping', 'Dyna-Q']
    
    num_of_mazes = 6
    
    mazes = [original_maze.extend_maze(i) for i in range(1, num_of_mazes + 1)]
    methods = [prioritized_sweeping, dyna_q]
    
    runs = 6
    all_backups = []

    # Use ProcessPoolExecutor to parallelize the runs
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_experiment, run, method_names, models, methods, params, mazes, num_of_mazes, params_dyna) for run in range(runs)]
        for future in futures:
            all_backups.append(future.result())

    # Compute the average over all runs
    backups = np.mean(all_backups, axis=0)

    # Dyna-Q performs several backups per step
    backups[1, :] *= params_dyna.planning_steps + 1

    for i in range(0, len(method_names)):
        plt.plot(np.arange(1, num_of_mazes + 1), backups[i, :], label=method_names[i])
    plt.xlabel('maze resolution factor')
    plt.ylabel('backups until optimal solution')
    plt.yscale('log')
    plt.legend()

    plt.savefig('./plots/example_8_4.png')
    plt.close()             


if __name__ == '__main__':
    example_8_4_parallel()        