import torch
import torch.nn as nn
import torch.optim as optim

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        """
        初始化Actor網絡。
        
        參數:
        state_dim (int): 狀態的維度。
        action_dim (int): 動作的維度。
        """
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),  # 輸入層到第一個隱藏層
            nn.ReLU(),  # 使用ReLU激活函數
            nn.Linear(128, 64),  # 第一個隱藏層到第二個隱藏層
            nn.ReLU(),  # 使用ReLU激活函數
            nn.Linear(64, action_dim),  # 第二個隱藏層到輸出層
            nn.Tanh()  # 使用Tanh激活函數來確保輸出值在[-1, 1]範圍內
        )
    
    def forward(self, state):
        """
        根據給定的狀態進行前向傳播，生成動作。
        
        參數:
        state (torch.Tensor): 當前智體的狀態。
        
        返回:
        torch.Tensor: 生成的動作。
        """
        return self.network(state)

'''
# 假設定義
state_dim = 24  # 單個智體狀態的維度
action_dim = 5  # 動作的維度

# 實例化Actor網絡
actor = Actor(state_dim, action_dim)

# 定義優化器
actor_optimizer = optim.Adam(actor.parameters(), lr=0.001)
'''

print("actor_done")