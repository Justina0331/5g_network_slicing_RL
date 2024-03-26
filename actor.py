import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        # 二元分類輸出
        self.binary_output = nn.Linear(64, 1)
        # 三元分類輸出
        self.trinary_output = nn.Linear(64, 3)
    
    def forward(self, state):
        features = self.network(state)
        # 使用sigmoid激活函數獲得二元分類的概率
        binary_prob = torch.sigmoid(self.binary_output(features))
        # 使用softmax激活函數獲得三元分類的概率分布
        trinary_probs = F.softmax(self.trinary_output(features), dim=0)

        # 生成介於 0 到 1 之間的隨機數
        random_val_bin = torch.rand(1).item()
        random_val_tri = torch.rand(1).item()

        # 根據隨機數的值決定是否進行隨機選擇
        if random_val_bin < 0.2:
            bin = 1 if binary_prob.item() > 0.5 else 0
        else:
            bin = 1 if binary_prob.item() <= 0.5 else 0

        if random_val_tri < 0.2:
            # 以 0.5 的機率隨機選擇 0 或 1 或 2
            tri = torch.randint(0, 3, (1,)).item()
        else:
            tri = torch.argmax(trinary_probs).item()

        tensor1 = torch.tensor(bin).unsqueeze(0)
        tensor2 = torch.tensor(tri).unsqueeze(0)
        action = torch.cat((tensor1, tensor2))

        #return action
        return tensor2
    
    
    '''
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
        return self.network(state)
    '''
    
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