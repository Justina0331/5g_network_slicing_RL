#import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import pandas as pd 
from collections import namedtuple, deque
#from tqdm import tqdm

from environment import Simulator
from actor import Actor
from critic import SingleCritic
from replayBuffer import ReplayBuffer
from classifierModel import classifierModel

# 條件設置
max_episodes = 3        # 迭代次數
batch_size = 1          # 抽取經驗大小
state_dim = 3           # 單個智體狀態的維度
action_dim = 5          # 動作的維度
sys_time = 10           # 訓練步數(模擬系統跑多久?)
max_alive_time = 100    # 封包最久處理多久?
sample_interval = 10    # 查看系統資源的時間間隔
amount = 10             # 每次迭代模擬的封包數量

# 定義Actor網路
actorA = Actor(state_dim, action_dim)
actor_optimizerA = optim.Adam(actorA.parameters(), lr=0.01)
actorB = Actor(state_dim, action_dim)
actor_optimizerB = optim.Adam(actorB.parameters(), lr=0.01)
actorC = Actor(state_dim, action_dim)
actor_optimizerC = optim.Adam(actorC.parameters(), lr=0.01)

# 定義Critic網路
n_agents = 3  # actor數量
critic = SingleCritic(state_dim, action_dim, n_agents)
critic_optimizer = optim.Adam(critic.parameters(), lr=0.001)

# 定義ReplayBuffer
buffer_capacity = 10000  # 緩衝區容量
replay_buffer = ReplayBuffer(capacity=buffer_capacity)

# 獲得封包資訊及分類模型
csv_path   = "data/Adjustment_v2.csv"
model_path = "models/rf.pickle"
packet_infos = classifierModel.get_infos_from_csv(csv_path)
model = classifierModel.get_model(model_path)

#訓練迭代過程
for episode in range(max_episodes):

    # 定義環境
    env = Simulator([20, 15, 10], packet_infos)
    env.reset()

    # 從模擬環境中產生多個模擬封包
    ready_packets = env.simulate(sys_time, max_alive_time, sample_interval, amount)

    # 這次迭代獲得的總獎勵
    episode_reward = 0

    for t in range(sys_time):

        state = env.get_state()             # 取得目前狀態
        action = actorA(state).detach()     # 根據當前狀態及策略選擇動作
        #print(action)

        # 在環境中執行動作
        next_state, reward, done, _ = env.step(action.numpy())  
        #print(next_state, reward, done)
        
        #replay_buffer.push(state, action, reward, next_state, done)  # 儲存經驗
        
        #if len(replay_buffer) > batch_size:
        #    experiences = replay_buffer.sample(batch_size)
            # 從經驗中提取資料
        #    states, actions, rewards, next_states, dones = experiences
            
            # 更新Critic網絡
            
            # 更新Actor網絡
            
            # 可能需要的其他步驟，如更新目標網絡等
            
        #state = next_state
        #if done:
        #    break

    print(f"Episode {episode+1}: Total Reward: {episode_reward}")
        


print("main_done")
