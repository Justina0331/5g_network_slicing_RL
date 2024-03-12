import torch
import torch.nn as nn
import torch.optim as optim

class SingleCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_agents):
        super(SingleCritic, self).__init__()
        # 調整輸入維度以包含所有智體的狀態和動作
        self.input_dim = (state_dim + action_dim) * n_agents
        self.output_dim = 1  # 價值函數的輸出是一個純量
        
        # 定義Critic網絡結構
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.output_dim)
        )
    
    def forward(self, state, action):
        # 將狀態和動作拼接作為輸入
        x = torch.cat([state, action], dim=1)
        return self.network(x)

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

print("critic_done")