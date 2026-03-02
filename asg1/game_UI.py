import tkinter as tk
from tkinter import ttk
import numpy as np
from core_game_env import MazeEnv
from Q_learning import QLearningAgent

class MazeGameUI:
    """Maze Game UI Interface"""
    def __init__(self, root):
        self.root = root
        self.root.title("Maze Game UI")
        self.root.resizable(False, False)

        #1. Initialize the environment and AI
        self.env = MazeEnv()
        self.agent = QLearningAgent(state_num=self.env.state_num, action_num=self.env.action_num)
        #Load the trained Q-table
        try:
            self.agent.q_table = np.load("q_table.npy")
            print("The Q table has been loaded successfully!")
        except:
            print("Warning: The Q-table file was not found. Please run train.py to complete the training first!")

        #2.UI configuration
        self.grid_size = 80
        self.canvas_width = self.env.grid_size * self.grid_size
        self.canvas_height = self.env.grid_size * self.grid_size

        #Color configuration
        self.color_bg = "#FFFFFF"
        self.color_wall = "#000000"
        self.color_start = "#90EE90"
        self.color_goal = "#FFD700"
        self.color_agent = "#1E90FF"
        self.color_grid = "#CCCCCC"

        #3.Create UI components
        self.main_frame = ttk.Frame(self.root, padding=10)
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        #Drawing a maze
        self.canvas = tk.Canvas(
            self.main_frame,
            width=self.canvas_width,
            height=self.canvas_height,
            bg=self.color_bg,
            highlightthickness=1,
            highlightbackground=self.color_grid
        )
        self.canvas.grid(row=0, column=0, rowspan=10, padx=(0, 20))

        #Displaying status, actions, rewards
        self.info_frame = ttk.LabelFrame(self.main_frame, text="game information", padding=10)
        self.info_frame.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=(0, 10))

        #Information label
        self.label_state = ttk.Label(self.info_frame, text="Current status: Starting point(0,0)")
        self.label_state.grid(row=0, column=0, sticky=tk.W, pady=2)

        self.label_action = ttk.Label(self.info_frame, text="Recent action: None")
        self.label_action.grid(row=1, column=0, sticky=tk.W, pady=2)

        self.label_reward = ttk.Label(self.info_frame, text="Reward：0")
        self.label_reward.grid(row=2, column=0, sticky=tk.W, pady=2)

        self.label_total_reward = ttk.Label(self.info_frame, text="Total rewards：0")
        self.label_total_reward.grid(row=3, column=0, sticky=tk.W, pady=2)

        self.label_step = ttk.Label(self.info_frame, text="Current step：0/20")
        self.label_step.grid(row=4, column=0, sticky=tk.W, pady=2)

        self.label_status = ttk.Label(self.info_frame, text="Game status: Waiting for start")
        self.label_status.grid(row=5, column=0, sticky=tk.W, pady=2)

        #Button panel
        self.btn_frame = ttk.Frame(self.main_frame, padding=10)
        self.btn_frame.grid(row=1, column=1, sticky=(tk.W, tk.E))

        self.btn_start = ttk.Button(
            self.btn_frame,
            text="START GAME",
            command=self.start_agent_run
        )
        self.btn_start.grid(row=0, column=0, padx=5, pady=5)

        self.btn_reset = ttk.Button(
            self.btn_frame,
            text="RESET GAME",
            command=self.reset_game
        )
        self.btn_reset.grid(row=0, column=1, padx=5, pady=5)

        #Marking running status
        self.is_running = False
        self.total_reward = 0

        #Draw maze
        self.draw_maze()

    #Draw the maze grid, obstacles, starting point, and ending point
    def draw_maze(self):
        self.canvas.delete("all")
        for x in range(self.env.grid_size):
            for y in range(self.env.grid_size):
                x1 = y * self.grid_size
                y1 = x * self.grid_size
                x2 = x1 + self.grid_size
                y2 = y1 + self.grid_size

                #Draw different types of grids
                if self.env.maze_map[x, y] == 1:
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill=self.color_wall, outline=self.color_grid)
                elif (x, y) == self.env.start_pos:
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill=self.color_start, outline=self.color_grid)
                elif (x, y) == self.env.end_pos:
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill=self.color_goal, outline=self.color_grid)
                else:
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill=self.color_bg, outline=self.color_grid)

        #Draw agent
        self.draw_agent()

    def draw_agent(self):
        x, y = self.env.current_pos
        padding = 10
        x1 = y * self.grid_size + padding
        y1 = x * self.grid_size + padding
        x2 = (y + 1) * self.grid_size - padding
        y2 = (x + 1) * self.grid_size - padding
        self.agent_id = self.canvas.create_oval(x1, y1, x2, y2, fill=self.color_agent, outline="")

    #Update the information panel
    def update_info(self, action="None", reward=0):
        x, y = self.env.current_pos
        self.label_state.config(text=f"Current status: Coordinates({x},{y})，Status Number{self.env._coord_to_state((x,y))}")
        self.label_action.config(text=f"Recent action：{action}")
        self.label_reward.config(text=f"Current reward：{reward}")
        self.label_total_reward.config(text=f"Total rewards：{self.total_reward}")
        self.label_step.config(text=f"Current steps：{self.env.current_step}/{self.env.max_step}")

    #Reset the game
    def reset_game(self):
        self.is_running = False
        self.env.reset()
        self.total_reward = 0
        self.draw_maze()
        self.update_info()
        self.label_status.config(text="Game status: Waiting for start")
        self.btn_start.config(state=tk.NORMAL)

    #Run the agent automatically 
    def start_agent_run(self):
        if self.is_running:
            return
        self.is_running = True
        self.btn_start.config(state=tk.DISABLED)
        self.label_status.config(text="Game status: Running")
        self.total_reward = 0
        self.run_step()

    def run_step(self):
        if not self.is_running:
            return

        #1.Retrieve the current status from the environment
        current_state = self.env._coord_to_state(self.env.current_pos)
        #2.Obtain the optimal action from the Q-learning algorithm
        action = self.agent.choose_best_action(current_state)
        #3.Execute the action in the environment and obtain the result
        next_state, reward, done, info = self.env.step(action)
        #Reward accumulation
        self.total_reward += reward

        #4. Update the UI and present visually
        self.draw_maze()
        self.update_info(action=info["action"], reward=reward)

        #Determine whether the game is over
        if done:
            self.is_running = False
            if info["is_goal"]:
                self.label_status.config(text="Game status: Successfully completed the game！")
            elif info["is_timeout"]:
                self.label_status.config(text="Game status: Timeout failed！")
            else:
                self.label_status.config(text="Game status: End of run!")
            self.btn_start.config(state=tk.NORMAL)
            return

        #Delay the execution of the next step by 0.3 seconds
        self.root.after(1000, self.run_step)

#Run the UI
if __name__ == "__main__":
    root = tk.Tk()
    app = MazeGameUI(root)
    root.mainloop()