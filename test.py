# -*- coding: utf-8 -*-
"""
使用 Sarsa 算法解决一个带障碍物的网格世界 (GridWorld) 寻路问题。
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from tqdm import tqdm
from collections import defaultdict


class GridWorld:
    """
    网格世界环境。

    定义了一个带有障碍物的 5x5 网格。智能体的目标是从起点 (0, 0)
    移动到终点 (3, 2)。
    """
    def __init__(self):
        """初始化网格世界环境。"""
        self.height = 5
        self.width = 5
        self.start_state = (0, 0)
        self.goal_state = (3, 2)

        # 定义网格中所有有效的、可移动的状态
        self.valid_states = {
            (0,0), (0,1), (0,2), (0,3), (0,4),
            (1,0),               (1,3), (1,4),
            (2,0), (2,1),        (2,3), (2,4),
            (3,0),        (3,2),        (3,4),
            (4,0),        (4,2), (4,3), (4,4)
        }

        # 将奖励值定义为常量，增强可读性
        self.REWARD_GOAL = 0
        self.REWARD_BOUNDARY = -10
        self.REWARD_STEP = -1

        # 定义动作空间 (0:上, 1:下, 2:左, 3:右)
        # 这是修复原代码的关键错误，Sarsa类需要这些属性
        self.actions = [0, 1, 2, 3]
        self.action_deltas = {
            0: (-1, 0),  # 上
            1: (1, 0),   # 下
            2: (0, -1),  # 左
            3: (0, 1),   # 右
        }
        self.action_symbols = {
            0: '↑', 1: '↓', 2: '←', 3: '→'
        }

    def step(self, state, action):
        """
        执行一个动作并返回环境的反馈。

        Args:
            state (tuple): 当前状态坐标 (row, col)。
            action (int): 要执行的动作。

        Returns:
            tuple: 一个元组 (next_state, reward, done)，分别表示
                   下一个状态、获得的奖励和回合是否结束。
        """
        delta = self.action_deltas[action]
        next_state = (state[0] + delta[0], state[1] + delta[1])
        done = False

        if next_state not in self.valid_states:
            # 如果移动到无效区域（墙或障碍物），则停留在原地并受到惩罚
            reward = self.REWARD_BOUNDARY
            next_state = state
        elif next_state == self.goal_state:
            # 如果到达终点，获得奖励，回合结束
            reward = self.REWARD_GOAL
            done = True
        else:
            # 正常移动，付出固定成本
            reward = self.REWARD_STEP

        return next_state, reward, done


class Sarsa:
    """
    Sarsa 算法智能体。

    使用 on-policy 的时序差分方法学习状态-动作价值函数 (Q-table)。
    """
    def __init__(self, env, epsilon, alpha, gamma):
        """
        初始化 Sarsa 智能体。

        Args:
            env (GridWorld): 智能体所在的环境。
            epsilon (float): epsilon-greedy策略中的探索率。
            alpha (float): 学习率。
            gamma (float): 折扣因子。
        """
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: np.zeros(len(self.env.actions)))

    def take_action(self, state):
        """
        根据 epsilon-greedy 策略选择一个动作。

        Args:
            state (tuple): 当前状态坐标。

        Returns:
            int: 选择的动作。
        """
        if np.random.random() < self.epsilon:
            # 以 epsilon 的概率进行探索
            return np.random.choice(self.env.actions)
        else:
            # 以 1-epsilon 的概率进行利用
            q_values = self.q_table[state]
            # 如果有多个Q值最大的动作，从中随机选择一个，避免策略固化
            return np.random.choice(np.where(q_values == np.max(q_values))[0])

    def update(self, s0, a0, r, s1, a1):
        """
        根据Sarsa公式: Q(s,a) <- Q(s,a) + alpha * (r + gamma*Q(s',a') - Q(s,a))
        来更新Q-table。

        Args:
            s0 (tuple): 初始状态。
            a0 (int): 在s0执行的动作。
            r (float): 获得的奖励。
            s1 (tuple): 下一个状态。
            a1 (int): 在s1将要执行的动作。
        """
        q_current = self.q_table[s0][a0]
        q_next = self.q_table[s1][a1]
        
        td_error = r + self.gamma * q_next - q_current
        self.q_table[s0][a0] += self.alpha * td_error


def visualize_policy(env, agent):
    """
    可视化学到的策略和从起点到终点的最优路径。

    Args:
        env (GridWorld): 网格世界环境。
        agent (Sarsa): 训练好的Sarsa智能体。
    """
    grid_data = np.zeros((env.height, env.width))
    for state in env.valid_states:
        grid_data[state] = 1  # 有效路径为白色
    grid_data[env.goal_state] = 2 # 目标为青色

    cmap = mcolors.ListedColormap(['#404040', 'white', 'cyan'])
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(grid_data, cmap=cmap, interpolation='nearest')

    # 1. 绘制策略：在每个状态格子上用箭头表示最优动作
    for state in env.valid_states:
        if state != env.goal_state:
            best_action = np.argmax(agent.q_table[state])
            arrow = env.action_symbols[best_action]
            ax.text(state[1], state[0], arrow,
                    ha='center', va='center', fontsize=20, color='black')

    # 2. 绘制路径：从起点开始，根据最优策略生成一条路径
    path = []
    current_state = env.start_state
    # 设置最大步数防止因策略不佳导致的无限循环
    for _ in range(env.width * env.height):
        path.append(current_state)
        if current_state == env.goal_state:
            break
        best_action = np.argmax(agent.q_table[current_state])
        next_state, _, _ = env.step(current_state, best_action)
        current_state = next_state
    
    if path and path[-1] == env.goal_state:
        rows, cols = zip(*path)
        ax.plot(cols, rows, color='red', linewidth=3, alpha=0.7)

    # 3. 标记起点和终点
    start_circle = plt.Circle((env.start_state[1], env.start_state[0]), 0.2, color='orange')
    goal_circle = plt.Circle((env.goal_state[1], env.goal_state[0]), 0.2, color='blue')
    ax.add_patch(start_circle)
    ax.add_patch(goal_circle)
    
    # 4. 设置网格和坐标轴样式
    ax.set_xticks(np.arange(-.5, env.width, 1), minor=True)
    ax.set_yticks(np.arange(-.5, env.height, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", size=0)
    ax.set_title("Sarsa Policy and Optimal Path", fontsize=16)
    plt.show()


def main():
    """主函数，负责执行整个Sarsa算法的训练和评估流程。"""
    # --- 1. 定义超参数 ---
    NUM_EPISODES = 500
    ALPHA = 0.1
    GAMMA = 0.9
    EPSILON = 0.1
    RANDOM_SEED = 0

    # --- 2. 初始化环境和智能体 ---
    np.random.seed(RANDOM_SEED)
    env = GridWorld()
    agent = Sarsa(env, epsilon=EPSILON, alpha=ALPHA, gamma=GAMMA)

    # --- 3. 训练循环（已简化） ---
    return_list = []
    length_list = []
    
    # 使用单个循环和tqdm来清晰地展示进度
    with tqdm(total=NUM_EPISODES, desc='Training Progress') as pbar:
        for i_episode in range(NUM_EPISODES):
            state = env.start_state
            action = agent.take_action(state)
            episode_return = 0
            episode_length = 0
            done = False
            
            while not done:
                next_state, reward, done = env.step(state, action)
                next_action = agent.take_action(next_state)
                
                agent.update(state, action, reward, next_state, next_action)
                
                state = next_state
                action = next_action
                episode_return += reward
                episode_length += 1
            
            return_list.append(episode_return)
            length_list.append(episode_length)
            
            # 更新进度条信息，显示最近10个回合的平均回报
            pbar.set_postfix({
                'episode': f'{i_episode + 1}/{NUM_EPISODES}',
                'avg_return': f'{np.mean(return_list[-10:]):.3f}'
            })
            pbar.update(1)

    # --- 4. 可视化结果 ---
    visualize_policy(env, agent)

    # 绘制回报曲线
    plt.figure(figsize=(10, 4))
    plt.plot(return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('Returns per Episode')
    plt.grid(True)
    plt.show()

    # 绘制回合长度曲线
    plt.figure(figsize=(10, 4))
    plt.plot(length_list)
    plt.xlabel('Episodes')
    plt.ylabel('Episode Length')
    plt.title('Episode Length per Episode')
    plt.grid(True)
    plt.show()


# 使用 if __name__ == '__main__': 是Python编程的最佳实践，
# 它可以防止在其他脚本导入此文件时执行训练代码。
if __name__ == '__main__':
    main()