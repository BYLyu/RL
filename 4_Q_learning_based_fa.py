import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from tqdm import tqdm

class GridWorld:
    def __init__(self):
        """初始化网格世界环境。"""
        self.height = 5
        self.width = 5
        self.start_state = (0, 0)
        self.goal_state = (3, 2)

        self.valid_states = {
            (0,0), (0,1), (0,2), (0,3), (0,4),
            (1,0),               (1,3), (1,4),
            (2,0), (2,1),        (2,3), (2,4),
            (3,0),        (3,2),        (3,4),
            (4,0),        (4,2), (4,3), (4,4)
        }
        # 显式定义障碍物/禁区
        self.forbidden_states = {
            (1, 1), (1, 2), (2, 2), (3, 1), (3, 3), (4, 1)
        }
        
        # 奖励设置
        self.REWARD_GOAL = 10.0
        self.REWARD_BOUNDARY = -1.0
        self.REWARD_FORBIDDEN = -1.0
        self.REWARD_STEP = -0.1

        # 定义动作空间 (↑, ↓, ←, →, •)
        self.actions = [0, 1, 2, 3, 4]
        self.action_deltas = {
            0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1), 4: (0, 0)
        }
        self.action_symbols = {
            0: '↑', 1: '↓', 2: '←', 3: '→', 4: '•'
        }

    def step(self, state, action):
        """
        执行一个动作并返回环境的反馈。

        Args:
            state (tuple): 当前状态坐标 (row, col)。
            action (int): 要执行的动作。

        Returns:
            tuple: (next_state, reward, done)
        """
        delta = self.action_deltas[action]
        next_state = (state[0] + delta[0], state[1] + delta[1])
        done = False

        if next_state == self.goal_state:
            reward = self.REWARD_GOAL
            done = True
        elif next_state in self.forbidden_states:
            reward = self.REWARD_FORBIDDEN
            next_state = state  # 撞到禁区则弹回
        elif next_state not in self.valid_states:
            reward = self.REWARD_BOUNDARY
            next_state = state  # 撞到边界则弹回
        else:
            reward = self.REWARD_STEP
        
        return next_state, reward, done


class QLearningFA:
    """
    使用线性函数近似的 Q-learning 智能体。
    """
    def __init__(self, env, epsilon, alpha, gamma):
        """
        初始化智能体。

        Args:
            env (GridWorld): 环境。
            epsilon (float): 探索率。
            alpha (float): 学习率。
            gamma (float): 折扣因子。
        """
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        # 基于特征向量的维度初始化权重矩阵
        self.feature_dim = len(self._get_features(self.env.start_state))
        self.weights = np.zeros((self.feature_dim, len(self.env.actions)))

    def _get_features(self, state):
        """
        将状态坐标转换为一个多项式特征向量。

        Args:
            state (tuple): 状态坐标。

        Returns:
            numpy.ndarray: 特征向量。
        """
        # 归一化坐标
        x = state[0] / (self.env.height - 1)
        y = state[1] / (self.env.width - 1)
        # 多项式特征
        return np.array([1.0, x, y, x**2, y**2, x*y, x**3, y**3, x**2*y, x*y**2])

    def predict_q(self, state):
        """
        使用当前权重预测给定状态下所有动作的Q值。

        Args:
            state (tuple): 状态坐标。

        Returns:
            numpy.ndarray: 该状态下所有动作的Q值数组。
        """
        features = self._get_features(state)
        return np.dot(features, self.weights)

    def take_action(self, state):
        """
        根据 epsilon-greedy 策略选择一个动作。
        """
        if np.random.random() < self.epsilon:
            return np.random.choice(self.env.actions)
        else:
            q_values = self.predict_q(state)
            # 增加一个检查，防止因数值问题出现NaN
            if np.isnan(q_values).any():
                return np.random.choice(self.env.actions)
            # 随机选择一个Q值最大的动作
            return np.random.choice(np.flatnonzero(q_values == q_values.max()))

    def update(self, s0, a0, r, s1, done): 
        """
        根据Q-learning和函数近似的规则更新权重。
        """
        q_current = self.predict_q(s0)[a0]
        
        q_next_max = 0
        if not done:
            q_next_max = np.max(self.predict_q(s1))
            
        td_target = r + self.gamma * q_next_max
        td_error = td_target - q_current
        
        # 更新权重：w <- w + alpha * td_error * ▽Q(s,a;w)
        # 对于线性函数近似, ▽Q(s,a;w) 就是特征向量 features
        features_s0 = self._get_features(s0)
        self.weights[:, a0] += self.alpha * td_error * features_s0


def visualize_policy(env, agent):
    """
    可视化学到的策略和最优路径。
    """
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
    for _ in range(env.width * env.height * 2): # 增加路径长度上限
        path.append(current_state)
        if current_state == env.goal_state: break
        
        q_values = agent.predict_q(current_state)
        if np.isnan(q_values).any(): break
            
        best_action = np.argmax(q_values)
        next_state, _, _ = env.step(current_state, best_action)
        
        # 防止在可视化路径时陷入循环
        if next_state in path: break
        current_state = next_state
        
    if path and path[-1] == env.goal_state:
        rows, cols = zip(*path)
        ax.plot(cols, rows, color='red', linewidth=3, marker='o', markersize=8, alpha=0.7)
        
    start_circle = plt.Circle((env.start_state[1], env.start_state[0]), 0.2, color='orange')
    goal_circle = plt.Circle((env.goal_state[1], env.goal_state[0]), 0.2, color='blue')
    ax.add_patch(start_circle)
    ax.add_patch(goal_circle)
    
    ax.set_xticks(np.arange(-.5, env.width, 1), minor=True)
    ax.set_yticks(np.arange(-.5, env.height, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", size=0)
    ax.set_title("Q-Learning (FA) Policy and Path", fontsize=16)
    plt.show()


def plot_learning_curves(return_list, length_list):
    """
    使用 subplot 将回报和回合长度的学习曲线绘制在一起。
    
    Args:
        return_list (list): 每个回合的总回报列表。
        length_list (list): 每个回合的步数列表。
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    # 绘制回报曲线
    axes[0].plot(return_list, label='Episode Return')
    axes[0].set_ylabel('Returns')
    axes[0].grid(True)
    
    # 绘制回合长度曲线
    axes[1].plot(length_list, label='Episode Length', color='orange')
    axes[1].set_xlabel('Episodes')
    axes[1].set_ylabel('Episode Length')
    axes[1].grid(True)

    plt.show()


def main():
    np.random.seed(0)
    env = GridWorld()

    config = {
        'num_episodes': 500,
        'max_steps_per_episode': 1000,
        'gamma': 0.9,
        'alpha_start': 0.1,
        'alpha_end': 0.01,
        'epsilon_start': 0.1,
        'epsilon_end': 0.01,
    }

    decay_episodes = config['num_episodes'] * 0.8
    config['alpha_decay'] = (config['alpha_start'] - config['alpha_end']) / decay_episodes
    config['epsilon_decay'] = (config['epsilon_start'] - config['epsilon_end']) / decay_episodes

    agent = QLearningFA(env, config['epsilon_start'], config['alpha_start'], config['gamma'])

    return_list = []
    length_list = []

    num_episodes = config['num_episodes']
    episodes_per_iteration = int(num_episodes / 10)

    for i in range(10):
        with tqdm(total=episodes_per_iteration, desc=f'Iteration {i+1}/10') as pbar:
            for i_episode_chunk in range(episodes_per_iteration):
                # 计算全局的回合索引，用于衰减 alpha 和 epsilon
                global_episode = i * episodes_per_iteration + i_episode_chunk
                
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

                # 使用全局回合索引来衰减 alpha 和 epsilon
                if global_episode < decay_episodes:
                    agent.alpha = max(config['alpha_end'], agent.alpha - config['alpha_decay'])
                    agent.epsilon = max(config['epsilon_end'], agent.epsilon - config['epsilon_decay'])

                # 使用全局回合索引来更新进度条后缀
                if (global_episode + 1) % 100 == 0: 
                    pbar.set_postfix({
                        'return': f'{np.mean(return_list[-100:]):.3f}',
                        'epsilon': f'{agent.epsilon:.3f}',
                        'length': f'{np.mean(length_list[-100:]):.1f}'
                    })
                pbar.update(1)

    # --- 可视化 ---
    visualize_policy(env, agent)
    plot_learning_curves(return_list, length_list)


if __name__ == '__main__':
    main()