import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from tqdm import tqdm
from collections import defaultdict

#网格世界环境
class GridWorld:
    def __init__(self):
        self.valid_states = {
            (0,0), (0,1), (0,2), (0,3), (0,4),
            (1,0),               (1,3), (1,4),
            (2,0), (2,1),        (2,3), (2,4),
            (3,0),        (3,2),        (3,4), 
            (4,0),        (4,2), (4,3), (4,4)
        }
        
        self.height = 5
        self.width = 5
        self.start_state = (0, 0)
        self.goal_state = (3, 2)

        # 定义奖励
        self.reward_target = 0
        self.reward_boundary = -10
        self.reward_other = -1

    def step(self, state, action):
        delta = self.action_deltas[action]
        next_state = (state[0] + delta[0], state[1] + delta[1])
        
        done = False

        if next_state in self.valid_states:
            if next_state == self.goal_state:
                reward = self.reward_target
                done = True 
            else:
                reward = self.reward_other
            return next_state, reward, done
        else:
            reward = self.reward_boundary
            return state, reward, done 

#Sarsa算法
class Sarsa:
    def __init__(self, env, epsilon, alpha, gamma):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: np.zeros(len(self.env.actions)))

    #动作选取，使用epsilon-greedy策略
    def take_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.env.actions)
        else:
            q_values = self.q_table[state]
            return np.random.choice(np.where(q_values == np.max(q_values))[0])

    def best_action(self, state):
        q_values = self.q_table[state]
        return np.argmax(q_values)

    # Sarsa算法
    def update(self, s0, a0, r, s1, a1):
        q_current = self.q_table[s0][a0]
        q_next = self.q_table[s1][a1]
        
        td_error = r + self.gamma * q_next - q_current
        self.q_table[s0][a0] += self.alpha * td_error

def visualize_policy(env, agent):
    grid_data = np.zeros((env.height, env.width))
    for state in env.valid_states:
        grid_data[state] = 1
    grid_data[env.goal_state] = 2

    cmap = mcolors.ListedColormap(['#606060', 'white', 'cyan'])
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(grid_data, cmap=cmap, interpolation='nearest')

    for state in env.valid_states:
        if state != env.goal_state:
            best_action_index = np.argmax(agent.q_table[state])
            arrow = env.action_symbols[best_action_index]
            ax.text(state[1], state[0], arrow,
                    ha='center', va='center', fontsize=20, color='black')

    path = []
    current_state = env.start_state
    
    for _ in range(env.width * env.height):
        path.append(current_state)
        if current_state == env.goal_state:
            break
        
        best_action = np.argmax(agent.q_table[current_state])
        delta = env.action_deltas[best_action]
        current_state = (current_state[0] + delta[0], current_state[1] + delta[1])
    
    if path and path[-1] == env.goal_state:
        rows, cols = zip(*path)
        ax.plot(cols, rows,  color='red', linewidth=3, markersize=10, alpha=0.7)

    circle1 = plt.Circle((env.start_state[1], env.start_state[0]), 0.1, color='orange', label='Start')
    circle2 = plt.Circle((env.goal_state[1], env.goal_state[0]), 0.1, color='blue', label='Goal')
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    
    ax.set_xticks(np.arange(env.width))
    ax.set_yticks(np.arange(env.height))
    ax.set_xticklabels(np.arange(1, env.width + 1))
    ax.set_yticklabels(np.arange(1, env.height + 1))
    ax.set_xticks(np.arange(-.5, env.width, 1), minor=True)
    ax.set_yticks(np.arange(-.5, env.height, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", size=0)
    plt.show()


env = GridWorld()
np.random.seed(0)
epsilon = 0.1
alpha = 0.1
gamma = 0.9

agent = Sarsa(env, epsilon, alpha, gamma)
num_episodes = 500

return_list = []
length_list = []

for i in range(10):
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):
            state = env.start_state
            action = agent.take_action(state)
            episode_return = 0
            episode_length = 0
            done = False
            
            while not done:
                next_state, reward, done = env.step(state, action)
                next_action = agent.take_action(next_state)
                episode_return += reward
                episode_length += 1 
                
                agent.update(state, action, reward, next_state, next_action)
                
                state = next_state
                action = next_action

            return_list.append(episode_return)
            length_list.append(episode_length)

            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({
                    'episode': f'{num_episodes / 10 * i + i_episode + 1}',
                    'return': f'{np.mean(return_list[-10:]):.3f}'
                })
            pbar.update(1)

episodes_list = list(range(len(return_list)))

visualize_policy(env, agent)

plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.show()


plt.plot(episodes_list, length_list) 
plt.xlabel('Episodes')
plt.ylabel('Episode Length')
plt.show()