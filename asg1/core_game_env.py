import numpy as np

class MazeEnv:
    """5×5 grid maze treasure hunt game environment"""
    def __init__(self):
        #Maze Basic Configuration
        self.grid_size = 5
        #0= blank space, 1= obstacle
        self.maze_map = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0]
        ])
        #Core points
        self.start_pos = (0, 0)
        self.end_pos = (4, 4)
        self.current_pos = self.start_pos
        self.current_step = 0
        self.max_step = 20  # Maximum steps in a single round

        #Definition of action space: 0 = up, 1 = down, 2 = left, 3 = right
        self.action_space = [0, 1, 2, 3]
        self.action_num = len(self.action_space)
        self.action_mapping = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        self.action_name = {0: "up", 1: "down", 2: "left", 3: "right"}

        #State space definition
        self.state_num = self.grid_size * self.grid_size
        self.init_state = self._coord_to_state(self.start_pos)

        #Reward function definition
        self.reward_goal = 100
        self.reward_hit = -10 
        self.reward_step = -1
        self.reward_timeout = -20

    #Coordinate transformation with a unique status number
    def _coord_to_state(self, pos):
        x, y = pos
        return x * self.grid_size + y

    #Returned status number to the coordinate
    def _state_to_coord(self, state):
        x = state // self.grid_size
        y = state % self.grid_size
        return (x, y)

    #Environment reset
    def reset(self):
        self.current_pos = self.start_pos
        self.current_step = 0
        return self._coord_to_state(self.current_pos)

    #Core interface
    def step(self, action):
        if action not in self.action_space:
            raise ValueError(f"Illegal action! It must be one of {self.action_space}")
        
        #Calculate new coordinates based on the movement
        current_x, current_y = self.current_pos
        dx, dy = self.action_mapping[action]
        new_x = current_x + dx
        new_y = current_y + dy

        #Initialization return value
        reward = 0
        done = False
        info = {
            "action": self.action_name[action],
            "is_hit": False,
            "is_goal": False,
            "is_timeout": False
        }

        #Boundary & Obstacle Detection
        if (new_x < 0 or new_x >= self.grid_size) or (new_y < 0 or new_y >= self.grid_size):
            #Exceed the grid boundary
            reward = self.reward_hit
            new_x, new_y = current_x, current_y
            info["is_hit"] = True
        elif self.maze_map[new_x, new_y] == 1:
            #Hit the obstacle
            reward = self.reward_hit
            new_x, new_y = current_x, current_y
            info["is_hit"] = True
        else:
            #Update location
            self.current_pos = (new_x, new_y)

        #Determinate destination
        if self.current_pos == self.end_pos:
            reward = self.reward_goal
            done = True
            info["is_goal"] = True
        else:
            self.current_step += 1
            #Overtime
            if self.current_step >= self.max_step:
                reward = self.reward_timeout
                done = True
                info["is_timeout"] = True
            elif not info["is_hit"]:
                reward = self.reward_step

        #Return the results to the Q-learning algorithm
        next_state = self._coord_to_state(self.current_pos)
        return next_state, reward, done, info

    #Print the state of the maze on the console
    def render(self):
        print(f"当前步数：{self.current_step}/{self.max_step}")
        print("-" * (self.grid_size * 2 + 1))
        for x in range(self.grid_size):
            line = "|"
            for y in range(self.grid_size):
                if (x, y) == self.current_pos:
                    line += "S|"  # S=agent
                elif (x, y) == self.end_pos:
                    line += "E|"  # E=destination
                elif self.maze_map[x, y] == 1:
                    line += "#|"  # #=obstacle
                else:
                    line += ".|"  # .=blank
            print(line)
            print("-" * (self.grid_size * 2 + 1))
        print("\n")