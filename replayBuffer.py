import torch
import numpy as np
import random
from collections import deque

import random

# 設置隨機種子
random.seed(62)
# 設置隨機種子
torch.manual_seed(42)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # 由於Tensor可能具有不同的形状，需要處理轉換
        states = torch.stack(states).numpy()
        actions = torch.stack(actions).numpy()
        rewards = np.array(rewards, dtype=np.float32)
        next_states = torch.stack(next_states).numpy()
        dones = np.array(dones, dtype=np.uint8)
        
        return states, actions, rewards, next_states, dones
    
    def sample_by_reward(self, n_samples):
        # 根據獎勵排序經驗，選擇前n_samples個高獎勵的經驗
        sorted_buffer = sorted(self.buffer, key=lambda x: x[2], reverse=True)
        high_reward_experiences = sorted_buffer[:n_samples]

        # 解包經驗並轉換為NumPy陣列
        states, actions, rewards, next_states, dones = zip(*high_reward_experiences)
        
        # 由於Tensor可能具有不同的形狀，需要處理轉換
        states = torch.stack(states).numpy()
        actions = torch.stack(actions).numpy()
        rewards = np.array(rewards, dtype=np.float32)
        next_states = torch.stack(next_states).numpy()
        dones = np.array(dones, dtype=np.uint8)
        
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

'''
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
    
    def push(self, state, action, reward, next_state, done):
        # 如果緩衝區已滿，則刪除獎勵最低的經驗
        if len(self.buffer) >= self.capacity:
            # 找到獎勵最低的經驗的索引
            min_reward_index = min(range(len(self.buffer)), key=lambda i: self.buffer[i][2])
            # 刪除獎勵最低的經驗
            del self.buffer[min_reward_index]
        # 添加新的經驗
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.stack(states).numpy()
        actions = torch.stack(actions).numpy()
        rewards = np.array(rewards, dtype=np.float32)
        next_states = torch.stack(next_states).numpy()
        dones = np.array(dones, dtype=np.uint8)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)
'''


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.position = 0

    def push(self, state, action, reward, next_state, done, priority):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
            self.priorities.append(None)
        # 直接將經驗以元組形式存儲
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.priorities[self.position] = priority ** self.alpha
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        scaled_priorities = np.array(self.priorities) / np.sum(self.priorities)
        sample_indices = np.random.choice(np.arange(len(self.buffer)), size=batch_size, p=scaled_priorities)
        
        weights = (1.0 / (len(self.buffer) * scaled_priorities[sample_indices])) ** beta
        weights /= weights.max()
        
        experiences = [self.buffer[idx] for idx in sample_indices]
        states = np.vstack([e[0] for e in experiences])
        actions = np.vstack([e[1] for e in experiences])
        rewards = np.vstack([e[2] for e in experiences])
        next_states = np.vstack([e[3] for e in experiences])
        dones = np.vstack([e[4] for e in experiences])

        return states, actions, rewards, next_states, dones#, weights, sample_indices

    def update_priorities(self, sample_indices, new_priorities):
        for idx, new_priority in zip(sample_indices, new_priorities):
            self.priorities[idx] = new_priority ** self.alpha
'''
# 實例化ReplayBuffer
# 假設我們的緩衝區容量為10000
buffer_capacity = 10000
replay_buffer = ReplayBuffer(capacity=buffer_capacity)

# 假設這是從環境中得到的一些經驗
state = np.random.randn(4)  # 假設狀態是一個4維向量
action = np.random.randint(0, 2)  # 假設動作是0或1
reward = 1  # 假設獎勵
next_state = np.random.randn(4)  # 假設下一個狀態也是一個4維向量
done = False  # 假設當前經驗並不是一個episode的結束

# 將經驗添加到ReplayBuffer
replay_buffer.push(state, action, reward, next_state, done)

# 當ReplayBuffer中有足夠多的經驗時，可以開始抽樣學習
if len(replay_buffer) > batch_size:
    batch_size = 32  # 假設我們每次從ReplayBuffer中抽取32個經驗進行學習
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
'''

#print("replayBuffer_done")