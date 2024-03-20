import torch
import numpy as np
import random
from collections import deque

'''
沒意外的話這個應該也沒什麼好改
'''
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