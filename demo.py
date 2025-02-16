import random  
  
# Define the environment  
class Environment:  
    def __init__(self, size=5):  
        self.size = size  
        self.reset()  
  
    def reset(self):  
        self.grid = [[' ' for _ in range(self.size)] for _ in range(self.size)]  
        self.user_pos = [0, 0]  
        self.agent_pos = [self.size - 1, self.size - 1]  
        # Initialize rewards with positive, zero, or negative values  
        self.rewards = [[random.choice([-1, 0, 1]) for _ in range(self.size)] for _ in range(self.size)]  
        self.rewards[self.user_pos[0]][self.user_pos[1]] = 0  
        self.rewards[self.agent_pos[0]][self.agent_pos[1]] = 0  
        return self.get_state('agent')  
  
    def get_state(self, entity):  
        pos = self.agent_pos if entity == 'agent' else self.user_pos  
        return tuple(pos)  
  
    def display(self):  
        print("\nEnvironment:")  
        for i in range(self.size):  
            row = ''  
            for j in range(self.size):  
                cell = '[  ]'  # Default empty cell  
                reward = self.rewards[i][j]  
                if reward == 1:  
                    cell = '[R ]'  # Cell with a positive reward  
                elif reward == -1:  
                    cell = '[B ]'  # Cell with a negative (bad) reward  
                if [i, j] == self.user_pos and [i, j] == self.agent_pos:  
                    cell = '[UA]'  
                elif [i, j] == self.user_pos:  
                    cell = '[U ]'  
                elif [i, j] == self.agent_pos:  
                    cell = '[ A]'  
                row += cell  
            print(row)  
        print("")  
  
    def update_position(self, entity, action):  
        pos = self.user_pos if entity == 'user' else self.agent_pos  
        old_pos = pos.copy()  
        if action == 'up' and pos[0] > 0:  
            pos[0] -= 1  
        elif action == 'down' and pos[0] < self.size - 1:  
            pos[0] += 1  
        elif action == 'left' and pos[1] > 0:  
            pos[1] -= 1  
        elif action == 'right' and pos[1] < self.size - 1:  
            pos[1] += 1  
        # else:  
            # Cannot move; stay in the same position  
  
        # Collect reward if any  
        reward = self.rewards[pos[0]][pos[1]]  
        if reward == 1:  
            print(f"{entity.capitalize()} found a positive reward at position {pos}!")  
            self.rewards[pos[0]][pos[1]] = 0  # Remove the reward after collection  
        elif reward == -1:  
            print(f"{entity.capitalize()} stepped on a bad reward at position {pos}.")  
            self.rewards[pos[0]][pos[1]] = 0  # Remove the bad reward after collection  
  
        # Check if collision occurs  
        if self.user_pos == self.agent_pos and entity == 'agent':  
            print("Agent and user collided!")  
  
        return reward  
  
# Agent implementing Q-learning  
class Agent:  
    def __init__(self, actions, alpha=0.5, gamma=0.9, epsilon=0.1):  
        self.actions = actions  # List of possible actions  
        self.q_table = {}       # Q-value table  
        self.alpha = alpha      # Learning rate  
        self.gamma = gamma      # Discount factor  
        self.epsilon = epsilon  # Exploration rate  
  
    def choose_action(self, state):  
        # Epsilon-greedy policy  
        if random.uniform(0, 1) < self.epsilon:  
            # Explore: choose a random action  
            action = random.choice(self.actions)  
        else:  
            # Exploit: choose the best known action  
            q_values = [self.get_q_value(state, a) for a in self.actions]  
            max_q = max(q_values)  
            # In case multiple actions have the same max Q-value, randomly choose among them  
            max_actions = [a for a, q in zip(self.actions, q_values) if q == max_q]  
            action = random.choice(max_actions)  
        return action  
  
    def learn(self, state, action, reward, next_state):  
        # Update Q-value using the Q-learning formula  
        current_q = self.get_q_value(state, action)  
        max_future_q = max([self.get_q_value(next_state, a) for a in self.actions])  
        new_q = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)  
        self.set_q_value(state, action, new_q)  
  
    def get_q_value(self, state, action):  
        # Return the Q-value for the given state-action pair  
        return self.q_table.get((state, action), 0.0)  
  
    def set_q_value(self, state, action, value):  
        self.q_table[(state, action)] = value  
  
def get_user_action():  
    action = input("Choose your action (up/down/left/right) or 'exit' to finish:\n")  
    while action.lower() not in ['up', 'down', 'left', 'right', 'exit']:  
        action = input("Invalid action. Please choose (up/down/left/right) or 'exit':\n")  
    return action.lower()  
  
def main():  
    env = Environment()  
    actions = ['up', 'down', 'left', 'right']  
    agent = Agent(actions)  
    user_total_reward = 0  
    agent_total_reward = 0  
    turn = 1  
  
    print("Welcome to the Collaborative RL Demo with Learning Agent!")  
    print("You and the agent will take turns moving in the environment.")  
    print("Collect rewards by moving over them. Avoid bad rewards. Type 'exit' to finish.\n")  
  
    env.display()  
  
    while True:  
        # User's turn  
        if turn % 2 != 0:  
            action = get_user_action()  
            if action == 'exit':  
                break  
            reward = env.update_position('user', action)  
            user_total_reward += reward  
            env.display()  
            print(f"Your total reward: {user_total_reward}\n")  
        else:  
            # Agent's turn  
            state = env.get_state('agent')  
            action = agent.choose_action(state)  
            print(f"Agent chooses to move {action}.")  
            reward = env.update_position('agent', action)  
            agent_total_reward += reward  
            next_state = env.get_state('agent')  
            agent.learn(state, action, reward, next_state)  
            env.display()  
            print(f"Agent's total reward: {agent_total_reward}\n")  
  
        turn += 1  
  
    print("Game over!")  
    print(f"Final rewards - You: {user_total_reward}, Agent: {agent_total_reward}")  
    print("Thank you for participating in the collaborative environment!")  
  
if __name__ == '__main__':  
    main()  
