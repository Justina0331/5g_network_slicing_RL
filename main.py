#import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
# import random
from collections import namedtuple, deque
#from tqdm import tqdm
import matplotlib.pyplot as plt

from environment import Simulator
from actor import Actor
from critic import SingleCritic
from replayBuffer import ReplayBuffer
from classifierModel import classifierModel

# 條件設置
max_episodes = 100        # 迭代次數
batch_size = 10          # 抽取經驗大小
state_dim = 8           # 單個智體狀態的維度
action_dim = 2          # 動作的維度
sys_time = 100           # 訓練步數(模擬系統跑多久?)
max_alive_time = 10    # 封包最久處理多久?
amount = 50             # 每次迭代模擬的封包數量
gamma = 0.7

# 定義Actor網路
actorA = Actor(state_dim, action_dim)
actor_optimizerA = optim.Adam(actorA.parameters(), lr=0.01)
actorB = Actor(state_dim, action_dim)
actor_optimizerB = optim.Adam(actorB.parameters(), lr=0.01)
actorC = Actor(state_dim, action_dim)
actor_optimizerC = optim.Adam(actorC.parameters(), lr=0.01)
actor = [actorA, actorB, actorC]

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

episode_reward = np.zeros(max_episodes)

#訓練迭代過程
for episode in range(max_episodes):

    # 定義環境
    env = Simulator([20, 10, 10], packet_infos)
    env.reset()
    state = env.get_state()             # 取得目前狀態

    # 從模擬環境中產生多個模擬封包
    env.simulate(model, sys_time, max_alive_time, amount)

    for t in range(sys_time):

        # 當前時間要釋放資源的封包
        if t in env.running_packets:
            for packet in env.running_packets[t]: 
                env.allocated_resources[packet.allocated_slice] -= 1
                
            del env.running_packets[t]
        
        # 當前時間要分配資源的封包
        if t in env.ready_packets:
            for packet in env.ready_packets[t]:
                # packet.allocated_slice：學長的分類器選擇要給哪個切片
                action = actor[packet.allocated_slice](state).detach()     # 根據當前狀態及策略選擇動作

                # 在環境中執行動作
                next_state, reward, done, _ = env.step(action.numpy(), packet) 
                episode_reward[episode] += reward
        
                replay_buffer.push(state, action, reward, next_state, done)  # 儲存經驗
                #print(state, action, reward, next_state, done)

                # 從經驗中更新網路
                if len(replay_buffer) >= batch_size:
                    # 從經驗中提取資料
                    states, actions, rewards, next_states, dones = replay_buffer.sample_by_reward(batch_size)
                    #print(states, actions, rewards, next_states, dones)

                    # 更新 Critic
                    critic.update(states, actions, next_states, rewards, critic_optimizer, gamma)

                    '''
                    這邊的muti-agent每個actor更新完全一樣
                    很怪
                    '''
                    # 更新 Actor
                    actor[0].update(states, actions, actor_optimizerA, critic)
                    actor[1].update(states, actions, actor_optimizerB, critic)
                    actor[2].update(states, actions, actor_optimizerC, critic)

                    # 可能需要的其他步驟，如更新目標網路等

                # state[slices_resource[3], total_sent_packets, rejected_packets, total_allocated_resources[3]]
                state = next_state
                
            del env.ready_packets[t]

    print(f"Episode {episode+1}: Total Reward: {round(episode_reward[episode], 2)}")
    if episode_reward[episode] > 0.8 * amount:
        break


# 創建一個折線圖
plt.figure(figsize=(10, 5))  # 圖片大小為10x5
plt.plot(episode_reward, marker='o', linestyle='-', color='b')

plt.title('Episode Rewards Over Time')  # 圖表標題
plt.xlabel('Episode')  # X軸標籤
plt.ylabel('Total Reward')  # Y軸標籤
plt.grid(True)  # 顯示網格線

# 顯示圖表
plt.show()





'''
# 併行處理封包
import multiprocessing

def process_packet(packet):
    # 这里是处理单个封包的逻辑
    # 例如：action = actor[packet.allocated_slice](state).detach()
    # next_state, reward, done, _ = env.step(action.numpy(), packet.allocated_slice)
    # 返回你需要的信息，例如：return next_state, reward, done
    pass

def handle_packets_at_time_t(packets):
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(process_packet, packets)
    return results

# 假设ready_packets[t]是包含当前时间步所有待处理封包的列表
if t in ready_packets:
    packets = ready_packets[t]
    results = handle_packets_at_time_t(packets)
    # 处理results，例如累计奖励等

'''