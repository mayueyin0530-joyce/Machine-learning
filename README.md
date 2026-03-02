# Machine-learning
The project uses Python to implement the 5×5 grid maze treasure hunt game based on Q-Learning. Q-learning is a key of machine learning. The core is that an agent learns the optimal strategy from reward signals by interacting with the environment. Therefore this model becomes the best choice for beginners to learn reinforcement learning due to its simple implementation and good convergence, and the discrete states and action space of the grid treasure hunt game are highly suitable for using the Q-learning algorithm.

Core_game_environment: This is the core of the maze environment. It configures the 5×5 maze, defines the map, core points, action and state spaces, provides conversion functions between coordinates and states, and core interfaces like reset() and step() for the agent. It also has a render() function to print the maze state on the console for debugging.

Q-Learning: The core of the agent that implements the Q-Learning algorithm and connects with “Core_game_environment” via standardized interfaces. It initializes parameters such as learning rate and exploration rate, and a Q-table. Meanwhile, the functions for action selection, Q-table update, exploration rate decay, and set a special function to select only the optimal action for UI display is designed.

Train: This part integrates the above two modules, it serves as the core training script of the project, which calls the “Core_game_environment” and “Q-Learning” to implement the full training process of the agent, and saves the trained Q-table for the UI module to call.

Game_UI: The UI is developed with Tkinter following the principle of simplicity and practicality. It is divided into three core areas: the maze canvas, the information panel, and the button panel. 
