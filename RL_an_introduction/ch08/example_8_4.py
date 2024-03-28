import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from maze import *

def example_8_4():
    original_maze = Maze()
    
    params_prioritized = DynaParams()
    params_prioritized.theta = 0.0001
    params_prioritized.planning_steps = 5
    params_prioritized.alpha = 0.5
    params_prioritized.gamma = 0.95
    
    params_dyna = DynaParams()
    params_dyna.planning_steps = 5
    params_dyna.alpha = 0.5
    params_dyna.gamma = 0.95
    
    params = [params_prioritized, params_dyna]
    models = [PriorityModel, TrivalModel]
    method_names = ['Prioritized Sweeping', 'Dyna-Q']
    
    num_of_mazes = 5
    
    mazes = [original_maze.extend_maze(i) for i in range(1, num_of_mazes + 1)]
    methods = [prioritized_sweeping, dyna_q]
    
    runs = 5
    
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
                
if __name__ == '__main__':
    example_8_4()        