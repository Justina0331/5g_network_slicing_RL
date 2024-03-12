import numpy as np
import random
from collections import namedtuple, deque

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), np.array(action), np.array(reward, dtype=np.float32), np.array(next_state), np.array(done, dtype=np.uint8)

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

print("replayBuffer_done")