import numpy as np
from gymnasium import Env
import pygame

class WindyGridworld(Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        king_move: bool = False,  # For exercise 6.9
        stochastic_wind: bool = False,  # For exercise 6.10
        render_mode: str = None,
        size: int = 40,
    ):
        self.cols = 10
        self.rows = 7
        self.start = (3, 0)
        self.goal = (3, 7)

        self.map = np.ones(shape=(self.rows, self.cols), dtype=int) * -1
        self.map[self.goal] = 0
        self.stochastic = stochastic_wind

        # define wind factors
        self.wind = np.zeros(shape=(self.cols,), dtype=int)
        self.wind[3:9] = 1
        self.wind[6:8] = 2

        self.nA = 4 if not king_move else 9
        self.nS = self.map.shape
        self.truncated = False

        self.act_move_map = {
            0: (-1, 0),  # UP
            1: (1, 0),  # DOWN
            2: (0, 1),  # RIGHT
            3: (0, -1),  # LEFT
            4: (-1, -1),  # UP LEFT
            5: (-1, 1),  # UP RIGHT
            6: (1, -1),  # DOWN LEFT
            7: (1, 1),  # DOWN RIGHT
            8: (0, 0),  # NO MOVE
        }

        # Initialize parameters for pygame
        self.size = size
        self.render_mode = render_mode
        self.window_size = (self.map.shape[1] * size, self.map.shape[0] * size)
        self.window = None  # window for pygame rendering
        self.clock = None  # clock for pygame ticks


    # Take an action and return a tuple of (S', R, Terminated)
    def step(self, action):
        terminated = False
        move = self.act_move_map[action]
        # Adding wind factor to the vertical move
        next_row = self.state[0] + move[0] - self.wind[self.state[1]]

        # Exercise 3.10, stochastic wind problem
        if self.stochastic and self.wind[self.state[1]]:
            prob = np.random.random(size=(1,))
            if prob <= 1 / 3:
                next_row -= 1
            elif prob <= 2 / 3:
                next_row += 1

        # check the row's bounds
        next_row = max(0, next_row)
        next_row = min(self.rows - 1, next_row)

        next_col = self.state[1] + move[1]
        next_col = max(0, next_col)
        next_col = min(self.cols - 1, next_col)

        next_state = (next_row, next_col)

        if next_state == self.goal:
            terminated = True
        
        self.state = next_state
        reward = self.map[next_state]

        if self.render_mode == "human":
            self.render(self.render_mode)
        
        return next_state, reward, terminated, self.truncated
    
    def reset(self):
        self.state = (3, 0)
        self.truncated = False
        if self.render_mode == "human":
            self.render(self.render_mode)
        return self.state
    
    def render(self, mode):
        if self.window is None:
            pygame.init()
            pygame.display.set_caption("Windy Gridworld")
            if mode == "human":
                self.window = pygame.display.set_mode(self.window_size)

        if self.clock is None:
            self.clock = pygame.time.Clock()

        size = self.size
        self.window.fill((255, 255, 255))
        # Draw the map
        for row in range(self.rows):
            for col in range(self.cols):
                fill = (120, 120, 120)
                if (row, col) == self.start:
                    fill = (235, 52, 52)
                    pygame.draw.rect(
                        surface=self.window,
                        color=fill,
                        rect=(col * size, row * size, size, size),
                        width=0,
                    )
                if (row, col) == self.goal:
                    fill = (61, 227, 144)
                    pygame.draw.rect(
                        surface=self.window,
                        color=fill,
                        rect=(col * size, row * size, size, size),
                        width=0,
                    )

                pygame.draw.rect(
                    surface=self.window,
                    color=fill,
                    rect=(col * size, row * size, size, size),
                    width=1,
                )
        # draw the agent
        pygame.draw.rect(
            surface=self.window,
            color=(86, 61, 227),
            rect=(self.state[1] * size, self.state[0] * size, size, size),
            width=0,
        )

        if mode == "human":
            pygame.display.update()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.window = None
                    pygame.quit()
                    self.truncated = True
            self.clock.tick(self.metadata["render_fps"])


if __name__ == "__main__":
    render_mode = None
    env = WindyGridworld(render_mode=render_mode, size=40)
    env.reset()
    ttl_reward = 0
    while True:
        action = np.random.choice(env.nA)
        next_state, reward, terminated = env.step(action)
        ttl_reward += reward
        if terminated:
            break
    print(ttl_reward)