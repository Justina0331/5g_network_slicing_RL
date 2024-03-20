import torch
import torch.nn as nn
import torch.optim as optim

class SingleCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_agents):
        super(SingleCritic, self).__init__()
        # 調整輸入維度以包含所有智體的狀態和動作
        self.input_dim = state_dim + action_dim
        # self.input_dim = (state_dim + action_dim) * n_agents
        self.output_dim = 1  # 價值函數的輸出是一個純量
        
        # 定義Critic網路結構
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.output_dim)
        )
    
    '''
    forward及上面的network是用來選擇動作，完全沒有設計過，可以改
    '''
    def forward(self, state, action):
        # 將狀態和動作拼接作為輸入
        x = torch.cat([state, action], dim=1)
        return self.network(x)
    
    '''
    更新critic神經網路，基本上也是沒有設計過，就只是能跑而已，也能改
    '''
    def update(self, states, actions, next_states, rewards, optimizer, gamma):
        # 將 states, actions, rewards, next_states 轉成 torch.Tensor
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(-1)
        next_states = torch.FloatTensor(next_states)

        # 計算當前Q值
        current_Q_values = self(states, actions)

        with torch.no_grad():
            # 計算下一個狀態的Q值
            next_Q_value = self(next_states, actions)
            # 計算目標Q值，簡化版，未考慮done標誌
            target_Q_value = rewards + gamma * next_Q_value

        # 計算損失函數
        critic_loss = nn.MSELoss()(current_Q_values, target_Q_value)

        # 梯度下降更新 Critic
        optimizer.zero_grad()
        critic_loss.backward()
        optimizer.step()


'''
# 假設定義
state_dim = 24  # 單個智體狀態維度
action_dim = 5  # 單個智體動作維度
n_agents = 3  # 智體數量

# 實例化單一Critic網絡
critic = SingleCritic(state_dim, action_dim, n_agents)

# 定義優化器
critic_optimizer = optim.Adam(critic.parameters(), lr=0.001)

# 以下是訓練過程的簡化版，展示如何使用單一Critic
'''

#print("critic_done")