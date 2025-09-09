import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from tqdm import tqdm

class GridWorld:
    """
    环境类
    # <--- 优化点: 更新了终点位置和奖励函数
    """
    def __init__(self):
        self.valid_states = {
            (0,0), (0,1), (0,2), (0,3), (0,4),
            (1,0),               (1,3), (1,4),
            (2,0), (2,1),         (2,3), (2,4),
            (3,0),        (3,2),        (3,4), 
            (4,0),        (4,2), (4,3), (4,4)
        }
        self.forbidden_states = {
            (1, 1), (1, 2),
            (2, 2),
            (3, 1), (3, 3),
            (4, 1)
        }
        
        self.height = 5
        self.width = 5
        self.start_state = (0, 0)
        # 更新终点到 "陷阱" 位置
        self.goal_state = (3, 2)

        # 重塑奖励函数以鼓励探索
        self.reward_target = 10.0      # 提高目标奖励
        self.reward_boundary = -1.0    # 降低撞墙惩罚
        self.reward_forbidden = -1.0   # 降低禁区惩罚
        self.reward_other = -0.1       # 保持每一步的轻微惩罚

        self.actions = [0, 1, 2, 3, 4] # ↑, ↓, ←, →, •
        self.action_deltas = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
        self.action_symbols = ['↑', '↓', '←', '→', '•']

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
        elif next_state in self.forbidden_states:
            reward = self.reward_forbidden
            return state, reward, done 
        else: # Boundary
            reward = self.reward_boundary
            return state, reward, done

class QLearningFA:
    def __init__(self, env, epsilon, alpha, gamma):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.feature_dim = len(self._get_features(self.env.start_state))
        self.weights = np.zeros((self.feature_dim, len(self.env.actions)))

    def _get_features(self, state):
        x = state[0] / (self.env.height - 1)
        y = state[1] / (self.env.width - 1)
        return np.array([1.0, x, y, x**2, y**2, x*y, x**3, y**3, x**2*y, x*y**2])

    def predict_q(self, state):
        features = self._get_features(state)
        return np.dot(features, self.weights)

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.env.actions)
        else:
            q_values = self.predict_q(state)
            if np.isnan(q_values).any():
                return np.random.choice(self.env.actions)
            return np.random.choice(np.flatnonzero(q_values == q_values.max()))

    def update(self, s0, a0, r, s1, done): 
        q_current = self.predict_q(s0)[a0]
        # --- Q-Learning 的核心 ---
        # 如果是终止状态，下一状态的价值为 0
        if done:
            q_next_max = 0
        else:
            # 离策略(Off-policy): 使用下一状态所有可能动作中最大的Q值来计算TD目标
            q_next_max = np.max(self.predict_q(s1))
        # -------------------------
        td_target = r + self.gamma * q_next_max
        td_error = td_target - q_current
        
        features_s0 = self._get_features(s0)
        self.weights[:, a0] += self.alpha * td_error * features_s0

def visualize_policy(env, agent):
    """可视化函数"""
    grid_data = np.zeros((env.height, env.width))
    for state in env.valid_states:
        grid_data[state] = 1
    grid_data[env.goal_state] = 2
    cmap = mcolors.ListedColormap(['#606060', 'white', 'cyan'])
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(grid_data, cmap=cmap, interpolation='nearest')
    for state in env.valid_states:
        if state != env.goal_state:
            q_values = agent.predict_q(state)
            if np.isnan(q_values).any(): continue
            best_action_index = np.argmax(q_values)
            arrow = env.action_symbols[best_action_index]
            ax.text(state[1], state[0], arrow, ha='center', va='center', fontsize=20, color='black')
    path = []
    current_state = env.start_state
    for _ in range(env.width * env.height * 2):
        path.append(current_state)
        if current_state == env.goal_state: break
        q_values = agent.predict_q(current_state)
        if np.isnan(q_values).any(): break
        best_action = np.argmax(q_values)
        delta = env.action_deltas[best_action]
        next_state = (current_state[0] + delta[0], current_state[1] + delta[1])
        if next_state not in env.valid_states or next_state in path: break
        current_state = next_state
    if path and path[-1] == env.goal_state:
        rows, cols = zip(*path)
        ax.plot(cols, rows,  color='red', linewidth=3, marker='o', markersize=8, alpha=0.7)
    circle1 = plt.Circle((env.start_state[1], env.start_state[0]), 0.2, color='orange', label='Start')
    circle2 = plt.Circle((env.goal_state[1], env.goal_state[0]), 0.2, color='blue', label='Goal')
    ax.add_patch(circle1); ax.add_patch(circle2)
    ax.set_xticks(np.arange(env.width)); ax.set_yticks(np.arange(env.height))
    ax.set_xticklabels(np.arange(1, env.width + 1)); ax.set_yticklabels(np.arange(1, env.height + 1))
    ax.set_xticks(np.arange(-.5, env.width, 1), minor=True); ax.set_yticks(np.arange(-.5, env.height, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", size=0)
    plt.show()

# --- 主程序 ---
env = GridWorld()
np.random.seed(0)

config = {
    'num_episodes': 500,
    'max_steps_per_episode': 1000,
    'gamma': 0.9,
    'alpha_start': 0.1,
    'alpha_end': 0.01,
    'epsilon_start': 0.1,
    'epsilon_end': 0.01,
}

# 调整Epsilon衰减策略，在更多回合内保持探索
decay_episodes = config['num_episodes'] * 0.8 # 在80%的回合中进行衰减
config['alpha_decay'] = (config['alpha_start'] - config['alpha_end']) / decay_episodes
config['epsilon_decay'] = (config['epsilon_start'] - config['epsilon_end']) / decay_episodes

agent = QLearningFA(env, config['epsilon_start'], config['alpha_start'], config['gamma'])

return_list = []
length_list = []
total_episodes = config['num_episodes']

with tqdm(total=total_episodes, desc='Training with Q-Learning') as pbar:
    for i_episode in range(total_episodes):
        state = env.start_state
        episode_return = 0
        episode_length = 0
        done = False

        while not done and episode_length < config['max_steps_per_episode']:
            action = agent.take_action(state)
            next_state, reward, done = env.step(state, action)
            
            agent.update(state, action, reward, next_state, done)
            
            state = next_state
            episode_return += reward
            episode_length += 1
        
        return_list.append(episode_return)
        length_list.append(episode_length)

        # 更新 alpha 和 epsilon
        if i_episode < decay_episodes:
            agent.alpha = max(config['alpha_end'], agent.alpha - config['alpha_decay'])
            agent.epsilon = max(config['epsilon_end'], agent.epsilon - config['epsilon_decay'])

        if (i_episode + 1) % 100 == 0: 
            pbar.set_postfix({
                'return': f'{np.mean(return_list[-100:]):.3f}',
                'epsilon': f'{agent.epsilon:.3f}',
                'length': f'{np.mean(length_list[-100:]):.1f}'
            })
        pbar.update(1)

# --- 绘图 ---
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