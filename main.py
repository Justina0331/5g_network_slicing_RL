#import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
# import random
from collections import namedtuple, deque
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy

from environment import Simulator
from actor import Actor
from critic import SingleCritic
from replayBuffer import ReplayBuffer, PrioritizedReplayBuffer
from classifierModel import classifierModel

# 條件設置
max_episodes = 10        # 迭代次數
batch_size = 64          # 抽取經驗大小
state_dim = 3           # 單個智體狀態的維度
action_dim = 1          # 動作的維度
sys_time = 10           # 訓練步數(模擬系統跑多久?)
max_alive_time = 5    # 封包最久處理多久?
amount = 30             # 每次迭代模擬的封包數量*3
gamma = 0.99

# 定義Actor網路
actorA = Actor(state_dim, action_dim)
actor_optimizerA = optim.Adam(actorA.parameters(), lr=0.1)
actorB = Actor(state_dim, action_dim)
actor_optimizerB = optim.Adam(actorB.parameters(), lr=0.1)
actorC = Actor(state_dim, action_dim)
actor_optimizerC = optim.Adam(actorC.parameters(), lr=0.1)
actor = [actorA, actorB, actorC]

# 定義Critic網路
n_agents = 3  # actor數量
critic = SingleCritic(state_dim, action_dim, n_agents)
critic_optimizer = optim.Adam(critic.parameters(), lr=0.1)

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

#訓練迭代過程
#for episode in tqdm(range(max_episodes)):
for episode in range(max_episodes):

    # 定義環境
    env = Simulator([5, 5, 5], packet_infos)
    env.reset()
    env.simulate(model, sys_time, max_alive_time, amount, y)

    for t in range(sys_time):

        '''
        if t not in env.running_packets and t not in env.ready_packets:
        # packet.allocated_slice：學長的分類器選擇要給哪個切片
        state = env.get_state(packet.allocated_slice, 0)             # 取得局部目前狀態
        action = actor[packet.allocated_slice](state)     # 根據當前狀態及策略選擇動作
        #action = actor[packet.allocated_slice](state).detach()     # 根據當前狀態及策略選擇動作
        # 在環境中執行動作
        next_state, reward, done, _ = env.step(action.numpy(), packet, 0) 

        replay_buffer.push(state, action, reward, next_state, done)  # 儲存經驗
        prioritized_replay_buffer.push(state, action, reward, next_state, done, episode)
        print(state, action, reward, next_state, done)

        # 從經驗中更新網路
        if len(replay_buffer) >= batch_size:
            # 從經驗中提取資料
            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
            #states, actions, rewards, next_states, dones = replay_buffer.sample_by_reward(batch_size)
            #states, actions, rewards, next_states, dones = prioritized_replay_buffer(batch_size)
            #print(states, actions, rewards, next_states, dones)
            
            # 更新 Critic
            critic.update(states, actions, next_states, rewards, critic_optimizer, gamma)
            # 更新 Actor
            actor[0].update(states, actions, actor_optimizerA, critic)
            actor[1].update(states, actions, actor_optimizerB, critic)
            actor[2].update(states, actions, actor_optimizerC, critic)

            # 可能需要的其他步驟，如更新目標網路等

        # state[slices_resource[3], total_sent_packets, rejected_packets, total_allocated_resources[3]]
        state = next_state
        '''
        
        # 當前時間要釋放資源的封包
        if t in env.running_packets:
            for packet in env.running_packets[t]: 
                env.allocated_resources[packet.allocated_slice] -= 1
            del env.running_packets[t]
        
        # 當前時間要分配資源的封包
        if t in env.ready_packets:
            for packet in env.ready_packets[t]:
                # packet.allocated_slice：學長的分類器選擇要給哪個切片
                state = env.get_state(packet.allocated_slice, 1)             # 取得局部目前狀態
                action = actor[packet.allocated_slice](state)     # 根據當前狀態及策略選擇動作
                #action = actor[packet.allocated_slice](state).detach()     # 根據當前狀態及策略選擇動作
                # 在環境中執行動作
                next_state, reward, done, _ = env.step(action.numpy(), packet, 1) 
                GSM, reject = env.performance_metrics()

                episode_reward[episode] += reward
                GSM_[episode] += GSM
                reject_[episode] += reject
        
                replay_buffer.push(state, action, reward, next_state, done)  # 儲存經驗
                prioritized_replay_buffer.push(state, action, reward, next_state, done, episode)
                print(state, action, reward, next_state, done)
                #print("\n")

                # 從經驗中更新網路
                if len(replay_buffer) >= batch_size:
                    # 從經驗中提取資料
                    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                    #states, actions, rewards, next_states, dones = replay_buffer.sample_by_reward(batch_size)
                    #states, actions, rewards, next_states, dones = prioritized_replay_buffer(batch_size)
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

    episode_num[episode] = env.total_sent_packets

    # -3~3, 0~1
    #GSM_[episode], reject_[episode] = env.performance_metrics()
    #episode_reward[episode] = reward
    print(episode+1, env.total_sent_packets, GSM_[episode]/env.total_sent_packets, reject_[episode]/env.total_sent_packets, episode_reward[episode]/env.total_sent_packets)
    #print("\n")
    #print("\n")

    #print(f"Episode {episode+1}: Total Reward: {round(episode_reward[episode], 2)}")
    #if episode_reward[episode] > 0.8 * amount:
    #    break



# 創建一個圖形，定義子圖的排列方式為一行三列
plt.figure(figsize=(20, 5))

# 第一個子圖：Episode GSM_
plt.subplot(1, 3, 1)
plt.plot(range(1, len(GSM_) + 1), GSM_ / episode_num, marker='o', linestyle='-', color='b')
plt.title('Episode GSM usage')  # 子圖標題
plt.xlabel('Episode')  # X軸標籤
plt.ylabel('Mean GSM_')  # Y軸標籤
plt.grid(True)  # 顯示網格線

# 第二個子圖：Episode Reject Rate
plt.subplot(1, 3, 2)
plt.plot(range(1, len(reject_) + 1), reject_ / episode_num, marker='o', linestyle='-', color='b')
plt.title('Episode Reject Rate')  # 子圖標題
plt.xlabel('Episode')  # X軸標籤
plt.ylabel('Mean Reject Rate')  # Y軸標籤
plt.grid(True)  # 顯示網格線

# 第三個子圖：Episode Rewards
plt.subplot(1, 3, 3)
plt.plot(range(1, len(episode_reward) + 1), episode_reward / episode_num, marker='o', linestyle='-', color='b')
plt.title('Episode Rewards')  # 子圖標題
plt.xlabel('Episode')  # X軸標籤
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