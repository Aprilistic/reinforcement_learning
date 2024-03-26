import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import heapq
from copy import deepcopy

class PriorityQueue:
    def __init__(self):
        self.pq = []
        self.entry_finder = {}
        self.REMOVED = '<removed-task>'
        self.counter = 0
        
    def add_item(self, item, priority=0):
        if item in self.entry_finder:
            self.remove_item(item)
        entry = [priority, self.counter, item]
        self.counter
        self.entry_finder[item] = entry
        heapq.heappush(self.pq, entry)
    
    def remove_item(self, item):
        entry = self.entry_finder.pop(item)
        entry[-1] = self.REMOVED
    
    def pop_item(self):
        while self.pq:
            priority, count, item = heapq.heappop(self.pq)
            if item is not self.REMOVED:
                del self.entry_finder[item]
                return item, priority
        raise KeyError('pop from an empty priority queue')
    
    def empty(self):
        return not self.entry_finder
    
class Maze:
    def __init__(self):
        self.WORLD_WIDTH = 9
        self.WORLD_HEIGHT = 6
        
        self.ACTION_UP = 0
        self.ACTION_DOWN = 1
        self.ACTION_LEFT = 2
        self.ACTION_RIGHT = 3
        self.actions = [self.ACTION_UP, self.ACTION_DOWN, self.ACTION_LEFT, self.ACTION_RIGHT]
        
        self.START_STATE = [2, 0]
        self.GOAL_STATES = [[0, 8]]
        
        self.obstacles = [[1, 2], [2, 2], [3, 2], [0, 7], [1, 7], [2, 7], [4, 5]]
        self.old_obstacles = None
        self.new_obstacles = None
        
        self.obstacle_switch_time = None
        
        # self.stateActionValues = np.zeros((self.WORLD_HEIGHT, self.WORLD_WIDTH, len(self.actions)))
        
        self.q_size = (self.WORLD_HEIGHT, self.WORLD_WIDTH, len(self.actions))
        
        self.max_steps = np.inf
        
        self.resolution = 1
        
    def extend_state(self, state, factor):
        new_state = [state[0] * factor, state[1] * factor]
        new_states = []
        for i in range(0, factor):
            for j in range(0, factor):
                new_states.append([new_state[0] + i, new_state[1] + j])
        return new_states
    
    def extend_maze(self, factor):
        new_maze = Maze()
        new_maze.WORLD_WIDTH = self.WORLD_WIDTH * factor
        new_maze.WORLD_HEIGHT = self.WORLD_HEIGHT * factor
        new_maze.START_STATE = [self.START_STATE[0] * factor, self.START_STATE[1] * factor]
        new_maze.GOAL_STATES = self.extend_state(self.GOAL_STATES[0], factor)
        new_maze.obstacles = []
        for state in self.obstacles:
            new_maze.obstacles.extend(self.extend_state(state, factor))
        new_maze.q_size = (new_maze.WORLD_HEIGHT, new_maze.WORLD_WIDTH, len(new_maze.actions))
        # new_maze.stateActionValues = np.zeros((new_maze.WORLD_HEIGHT, new_maze.WORLD_WIDTH, len(new_maze.actions)))
        new_maze.resolution = factor
        return new_maze

    def step(self, state, action):
        x, y = state
        if action == self.ACTION_UP:
            x = max(x - 1, 0)
        elif action == self.ACTION_DOWN:
            x = min(x + 1, self.WORLD_HEIGHT - 1)
        elif action == self.ACTION_LEFT:
            y = max(y - 1, 0)
        elif action == self.ACTION_RIGHT:
            y = min(y + 1, self.WORLD_WIDTH - 1)
        if [x, y] in self.obstacles:
            x, y = state
        if [x, y] in self.GOAL_STATES:
            reward = 1.0
        else:
            reward = 0.0
        return [x, y], reward
    
class DynaParams:
    def __init__(self):
        self.gamma = 0.95
        self.epsilon = 0.1
        self.alpha = 0.1
        self.time_weight = 0
        self.planning_steps = 5
        self.runs = 10
        self.methods = ['Dyna-Q', 'Dyna-Q+']
        self.theta = 0
        
def choose_action(state, q_value, maze, dyna_params):
    if np.random.binomial(1, dyna_params.epsilon) == 1:
        return np.random.choice(maze.actions)
    else:
        values = q_value[state[0], state[1], :]
        return np.random.choice([action for action, value in enumerate(values) if value == np.max(values)])

# dyna-q
class TrivalModel:
    def __init__(self, rand = np.random):
        self.model = dict()
        self.rand = rand
        
    def feed(self, state, action, next_state, reward):
        state = deepcopy(state)
        next_state = deepcopy(next_state)
        if tuple(state) not in self.model.keys():
            self.model[tuple(state)] = dict()
        self.model[tuple(state)][action] = [list(next_state), reward]
        
    def sample(self):
        state_index = self.rand.choice(range(len(self.model.keys())))
        state = list(self.model)[state_index]
        action_index = self.rand.choice(range(len(self.model[state].keys())))
        action = list(self.model[state])[action_index]
        next_state, reward = self.model[state][action]
        state = deepcopy(state)
        next_state = deepcopy(next_state)
        return list(state), action, list(next_state), reward

# dyna-q+
class TimeModel:
    def __init__(self, maze, time_weight = 1e-4, rand=np.random):
        self.rand = rand
        self.model = dict()
        
        self.time = 0
        self.time_weight = time_weight
        self.maze = maze
        
    def feed(self, state, action, next_state, reward):
        state = deepcopy(state)
        next_state = deepcopy(next_state)
        self.time += 1
        
        if tuple(state) not in self.model.keys():
            self.model[tuple(state)] = dict()
            
            # Actions that had never been tried before were allowed in the planning
            for action_ in self.maze.actions:
                if action_ != action:
                    # Such actions' reward = 0, time = 1
                    self.model[tuple(state)][action_] = [list(state), 0, 1]
        
        self.model[tuple(state)][action] = [list(next_state), reward, self.time]
    
    def sample(self):
        state_index = self.rand.choice(range(len(self.model.keys())))
        state = list(self.model)[state_index]
        action_index = self.rand.choice(range(len(self.model[state].keys())))
        action = list(self.model[state])[action_index]
        next_state, reward, time = self.model[state][action]
        
        # adjust reward with elapsed time since last visit
        reward += self.time_weight * np.sqrt(self.time - time)
        
        state = deepcopy(state)
        next_state = deepcopy(next_state)
        return list(state), action, list(next_state), reward
        
        
def dyna_q(q_value, model, maze, dyna_params):
    state = maze.START_STATE
    steps = 0
    while state not in maze.GOAL_STATES:
        steps += 1
        action = choose_action(state, q_value, maze, dyna_params)
        next_state, reward = maze.step(state, action)
        
        q_value[state[0], state[1], action] += \
            dyna_params.alpha * (reward + dyna_params.gamma * np.max(q_value[next_state[0], next_state[1], :]) - q_value[state[0], state[1], action])
        
        model.feed(state, action, next_state, reward)            
        
        for t in range(0, dyna_params.planning_steps):
            state_, action_, next_state_, reward_ = model.sample()
            q_value[state_[0], state_[1], action_] += \
                dyna_params.alpha * (reward_ + dyna_params.gamma * np.max(q_value[next_state_[0], next_state_[1], :]) - q_value[state_[0], state_[1], action_])
                
        state = next_state
        
        if steps > maze.max_steps:
            break
    
    return steps

def changing_maze(maze, dyna_params):
    max_steps = maze.max_steps
    
    # cumulative rewards
    rewards = np.zeros((dyna_params.runs, 2, max_steps))
    
    for run in tqdm(range(dyna_params.runs)):
        models = [TrivalModel(), TimeModel(maze, time_weight=dyna_params.time_weight)]
        
        q_values = [np.zeros(maze.q_size), np.zeros(maze.q_size)]
        
        for i in range(len(dyna_params.methods)):
            maze.obstacles = maze.old_obstacles
            
            steps = 0
            last_steps = steps
            while steps < max_steps:
                steps += dyna_q(q_values[i], models[i], maze, dyna_params)
                
                rewards[run, i, last_steps: steps] = rewards[run, i, last_steps]
                rewards[run, i, min(steps, max_steps - 1)] = rewards[run, i, last_steps] + 1
                last_steps = steps
                
                if steps > maze.obstacle_switch_time:
                    maze.obstacles = maze.new_obstacles
                    
    rewards = rewards.mean(axis=0)
    
    return rewards