from tqdm import tqdm
import numpy as np
import torch
import collections
import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity) 

    def add(self, state, action, reward, next_state, done): 
        self.buffer.append((state, action, reward, next_state, done)) 

    def sample(self, batch_size): 
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done 

    def size(self): 
        return len(self.buffer)

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

def train_on_policy_agent(env, agent, num_episodes):
    """
    修正了循环范围和tqdm显示逻辑的最终版本
    """
    return_list = []
    # 外层循环，将总回合数分为10次迭代显示
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc=f'Iteration {i}') as pbar:
            # [修正1] 内层循环的范围应为总数的 1/10
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                transition_dict = {
                    'states': [],
                    'actions': [],
                    'next_states': [],
                    'rewards': [],
                    'dones': []
                }
                
                state, info = env.reset()
                done = False
                
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated

                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                
                return_list.append(episode_return)
                agent.update(transition_dict)

                if (i_episode + 1) % 10 == 0:
                    # [修正2] 更新postfix的逻辑，以显示正确的总回合数
                    current_episode_num = num_episodes // 10 * i + i_episode + 1
                    pbar.set_postfix({
                        'episode': f'{current_episode_num}',
                        'return': f'{np.mean(return_list[-10:]):.3f}'
                    })
                pbar.update(1)
                
    return return_list


def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    """ 
    修正了兼容性和逻辑错误的 off-policy 训练函数 
    """
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc=f'Iteration {i}') as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                # [修正1] 正确处理 gymnasium 的 reset() 返回值
                state, info = env.reset()
                done = False
                
                while not done:
                    action = agent.take_action(state)
                    
                    # [修正2] 正确处理 gymnasium 的 step() 返回值
                    next_state, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated

                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    
                    # 当经验回放池中数据足够时，才开始训练
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {
                            'states': b_s, 
                            'actions': b_a, 
                            'next_states': b_ns, 
                            'rewards': b_r, 
                            'dones': b_d
                        }
                        agent.update(transition_dict)

                # [修正3] 将记录回报的逻辑移到 while 循环外部
                return_list.append(episode_return)
                
                if (i_episode + 1) % 10 == 0:
                    # 更新进度条显示逻辑
                    current_episode_num = num_episodes // 10 * i + i_episode + 1
                    pbar.set_postfix({
                        'episode': f'{current_episode_num}',
                        'return': f'{np.mean(return_list[-10:]):.3f}'
                    })
                pbar.update(1)
                
    return return_list


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)