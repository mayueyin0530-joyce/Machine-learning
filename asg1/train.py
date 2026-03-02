from core_game_env import MazeEnv
from Q_learning import QLearningAgent
import numpy as np
import matplotlib.pyplot as plt

#1.Initialize the environment and the agent
env = MazeEnv()
agent = QLearningAgent(state_num=env.state_num, action_num=env.action_num)

#2.Training configuration
train_episodes = 1000 
reward_list = []

#3.Training loop
print("START")
for episode in range(train_episodes):
    #Reset the environment and obtain the initial state
    state = env.reset()
    total_reward = 0
    done = False

    #Single-round game
    while not done:
        #AI selects the action
        action = agent.choose_action(state)
        #Environmental action execution
        next_state, reward, done, info = env.step(action)
        #AI updates the Q-table
        agent.update_q_table(state, action, reward, next_state)
        #Cumulative rewards
        total_reward += reward
        #Update status
        state = next_state

    #After each round, the exploration rate is reduced.
    agent.epsilon_decay()
    reward_list.append(total_reward)

    #Printing training progress every 100 rounds
    if (episode + 1) % 100 == 0:
        print(f"Training Rounds:{episode+1}/{train_episodes}. This round's cumulative reward:{total_reward}. Current exploration rate:{agent.epsilon:.3f}")

print("TRAINING FINISHED")

#4.Save the trained Q-table
np.save("q_table.npy", agent.q_table)
print("The Q table has been saved as q_table.npy")

#5.Draw the training reward curve
plt.figure(figsize=(10, 6))
plt.plot(reward_list)
plt.xlabel("Training Round")
plt.ylabel("Cumulative rewards")
plt.title("Q-Learning Training reward variation curve")
plt.grid(True)
plt.savefig("train_curve.png")
plt.show()

# 6. Test the effectiveness of the trained AI
print("\n Test the effectiveness of the trained AI")
test_episodes = 10
success_count = 0
for episode in range(test_episodes):
    state = env.reset()
    done = False
    step_count = 0
    while not done:
        action = agent.choose_best_action(state)
        next_state, reward, done, info = env.step(action)
        state = next_state
        step_count += 1
    if info["is_goal"]:
        success_count += 1
        print(f"Test round{episode+1}：Successfully completed the level, steps:{step_count}")
    else:
        print(f"Test round{episode+1}：Failed to completed the level.")

print(f"\n Test completed. Pass rate:{success_count/test_episodes*100}%")