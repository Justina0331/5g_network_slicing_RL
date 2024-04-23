import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
# import random
from collections import namedtuple, deque
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy
import multiprocessing

from environment import Simulator
#from actor import Actor
#from critic import SingleCritic
from replayBuffer import ReplayBuffer, PrioritizedReplayBuffer
from classifierModel import classifierModel
from MADDPG import MADDPG

# 設置隨機種子
torch.manual_seed(42)

# 條件設置
max_episodes = 1000        # 迭代次數
batch_size = 180          # 抽取經驗大小
state_dim = 3           # 單個智體狀態的維度
action_dim = 1          # 動作的維度
sys_time = 10           # 訓練步數(模擬系統跑多久?)
max_alive_time = 5    # 封包最久處理多久?
amount = 30             # 每次迭代模擬的封包數量*3
gamma = 0.9
n_agents = 3  # actor數量
actor_lr = 0.0001
critic_lr = 0.00001

# 定義MADDPG
maddpg = MADDPG(state_dim, action_dim, n_agents, actor_lr, critic_lr, gamma)


# 定義ReplayBuffer
buffer_capacity = 10000  # 緩衝區容量
replay_buffer = ReplayBuffer(capacity=buffer_capacity)
prioritized_replay_buffer = PrioritizedReplayBuffer(capacity=buffer_capacity)

# 獲得封包資訊及分類模型
csv_path   = "data/Adjustment_v2.csv"
model_path = "models/rf.pickle"
packet_infos, y = classifierModel.get_infos_from_csv(csv_path)
model = classifierModel.get_model(model_path)

episode_reward = np.zeros(max_episodes)
GSM_ = np.zeros(max_episodes)
reject_ = np.zeros(max_episodes)
episode_num = np.zeros(max_episodes)
loss = np.zeros(max_episodes)

'''
# 定義環境
env = Simulator([5, 5, 5], packet_infos)
pa = env.simulate(model, sys_time, max_alive_time, amount, y)
p = copy.copy(pa)
env.reset(p)
'''

def process_packet(t, packet, receive, episode):
    state = env.get_state(packet.allocated_slice, receive)             # 取得局部目前狀態
    action = maddpg.actors[packet.allocated_slice](state, episode)     # 根據當前狀態及策略選擇動作
    #action = actor[packet.allocated_slice](state).detach()     # 根據當前狀態及策略選擇動作
    # 在環境中執行動作
    next_state, reward, done, _ = env.step(action.numpy(), packet, receive) 
    episode_reward[episode] += reward

    replay_buffer.push(state, action, reward, next_state, done)  # 儲存經驗
    #prioritized_replay_buffer.push(state, action, reward, next_state, done, episode)

    #print(t, packet.allocated_slice, state, action, reward, next_state, done)

episode_count =  0

#訓練迭代過程
#for episode in tqdm(range(max_episodes)):
for episode in range(max_episodes):
    episode_count += 1
    # 定義環境
    env = Simulator([5, 5, 5], packet_infos)
    pa = env.simulate(model, sys_time, max_alive_time, amount, y)
    #p = copy.copy(pa)
    env.reset(pa)

    for t in range(sys_time):

        # 當前時間要釋放資源的封包
        if t in env.running_packets:
            for packet in env.running_packets[t]: 
                env.allocated_resources[packet.allocated_slice] -= 1
                process_packet(t, packet, 0, episode)
            del env.running_packets[t]
        
        # 當前時間要分配資源的封包
        if t in env.ready_packets:
            for packet in env.ready_packets[t]:
                process_packet(t, packet, 1, episode)

            del env.ready_packets[t]

        # 從經驗中更新網路
        if len(replay_buffer) >= batch_size:
            # 從經驗中提取資料
            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
            #states, actions, rewards, next_states, dones = replay_buffer.sample_by_reward(batch_size)
            #states, actions, rewards, next_states, dones = prioritized_replay_buffer.sample(batch_size)
            #print(states, actions, rewards, next_states, dones)

            # 更新神經網路
            loss[episode] += maddpg.update_agents(states, actions, next_states, rewards, dones)


            # 可能需要的其他步驟，如更新目標網路等


                
    GSM, reject= env.performance_metrics()

    GSM_[episode] = GSM
    reject_[episode] = reject
    episode_num[episode] = env.total_sent_packets

    print(episode+1, env.total_sent_packets, "{:.2f}".format(loss[episode] / sys_time), "{:.2f}".format(GSM_[episode]), "{:.2f}".format(reject_[episode]), 
          "{:.2f}".format(episode_reward[episode]/env.total_sent_packets))
    
    if episode > 10 and loss[episode] / sys_time < 0.001:
        break

# 存取模型
for i, actor in enumerate(maddpg.actors):
    torch.save(actor.state_dict(), f'actor_model_{i}.pth')
torch.save(maddpg.critic.state_dict(), 'critic_model.pth')


# 創建一個圖形，定義子圖的排列方式為一行三列
plt.figure(figsize=(10, 6))

# 第一個子圖：Episode loss
plt.subplot(2, 2, 1)
plt.plot(range(5, episode_count + 1), loss[4:episode_count] / sys_time, marker='o', linestyle='-', color='b')
plt.title('Episode critic loss')  # 子圖標題
#plt.xlabel('Episode')  # X軸標籤
plt.ylabel('Mean loss')  # Y軸標籤
plt.grid(True)  # 顯示網格線


# 第一個子圖：Episode GSM_
plt.subplot(2, 2, 2)
plt.plot(range(1, episode_count + 1), GSM_[:episode_count], marker='o', linestyle='-', color='b')
plt.title('Episode GSM usage')  # 子圖標題
#plt.xlabel('Episode')  # X軸標籤
plt.ylabel('Mean GSM_')  # Y軸標籤
plt.grid(True)  # 顯示網格線


# 第二個子圖：Episode Reject Rate
plt.subplot(2, 2, 3)
plt.plot(range(1, episode_count + 1), reject_[:episode_count], marker='o', linestyle='-', color='b')
plt.title('Episode Reject Rate')  # 子圖標題
#plt.xlabel('Episode')  # X軸標籤
plt.ylabel('Mean Reject Rate')  # Y軸標籤
plt.grid(True)  # 顯示網格線

'''
# 第三個子圖：Episode Rewards
plt.subplot(2, 2, 4)
plt.plot(range(1, episode_count + 1), episode_reward[:episode_count] / episode_num[:episode_count], marker='o', linestyle='-', color='b')
plt.title('Episode Rewards')  # 子圖標題
#plt.xlabel('Episode')  # X軸標籤
plt.ylabel('Mean Reward')  # Y軸標籤
plt.grid(True)  # 顯示網格線
'''

# 第三個子圖：Episode Rewards
plt.subplot(2, 2, 4)
plt.plot(range(1, episode_count + 1), episode_reward[:episode_count] / episode_num[:episode_count], marker='o', linestyle='-', color='b')
plt.title('Episode Rewards')  # 子圖標題
#plt.xlabel('Episode')  # X軸標籤
plt.ylabel('Mean Reward')  # Y軸標籤
plt.grid(True)  # 顯示網格線

# 顯示圖形
plt.show()




'''
# 創建一個折線圖
plt.figure(figsize=(10, 5))  # 圖片大小為10x5
plt.plot(sys_reward, marker='o', linestyle='-', color='b')

plt.title('Episode Rewards Over Time')  # 圖表標題
plt.xlabel('Episode')  # X軸標籤
plt.ylabel('Total Reward')  # Y軸標籤
plt.grid(True)  # 顯示網格線

# 顯示圖表
plt.show()
'''


