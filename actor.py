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
    
    '''
    forward及上面的network是用來選擇動作，完全沒有設計過，可以改
    '''
    def forward(self, state):
        return self.network(state)
    
    '''
    更新actor神經網路，基本上也是沒有設計過，就只是能跑而已，也能改
    '''
    def update(self, states, actions, optimizer, critic):
        # 將 states, actions, rewards, next_states 轉成 torch.Tensor
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)

        # 計算actor損失函數
        actor_loss = -critic(states, actions).mean()

        # 更新 Actor
        optimizer.zero_grad()
        actor_loss.backward()
        optimizer.step()

'''
# 假設定義
state_dim = 24  # 單個智體狀態的維度
action_dim = 5  # 動作的維度

# 實例化Actor網絡
actor = Actor(state_dim, action_dim)

# 定義優化器
actor_optimizer = optim.Adam(actor.parameters(), lr=0.001)
'''


'''
# 將action輸出設置在0~1之間
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, input_size, output_size):
        super(Actor, self).__init__()
        # 定义网络结构
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
            nn.Sigmoid()  # 使用Sigmoid激活函数确保输出在0到1之间
        )
        
    def forward(self, x):
        return self.network(x

'''