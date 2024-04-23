import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import copy
import random

# 設置隨機種子
random.seed(2)
# 設置隨機種子
torch.manual_seed(78)

class Actor(nn.Module):
    
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 4),
            nn.ReLU(),
            #nn.Linear(32, 16),
            #nn.ReLU(),
        )
        self.trinary_output = nn.Linear(4, 3)
    
    def forward(self, state, iteration):
        features = self.network(state)
        #if features.dim() == 1:
        #    features = features.unsqueeze(0)  # 將一維張量轉換為二維張量
        #trinary_probs = F.softmax(self.trinary_output(features), dim=1)
        trinary_probs = F.softmax(self.trinary_output(features), dim=0)
        random_val_tri = torch.rand(1).item()

        exploration_prob = max(0, 1.0 - iteration * 0.01)

        if random_val_tri < exploration_prob:
            # 以 0.1 的機率隨機選擇 0 或 1 或 2
            tri = torch.randint(0, 3, (1,)).item()
        else:
            tri = torch.argmax(trinary_probs).item()

        tensor = torch.tensor(tri).unsqueeze(0)

        return tensor
    
class SingleCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_agents):
        super(SingleCritic, self).__init__()
        # 調整輸入維度以包含所有智體的狀態和動作
        self.input_dim = state_dim + action_dim
        self.output_dim = 1  # 價值函數的輸出是一個純量
        
        # 定義Critic網路結構
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, 4),
            nn.ReLU(),
            #nn.Linear(32, 16),
            #nn.ReLU(),
            nn.Linear(4, self.output_dim)
        )
    
    def forward(self, state, action):
        # 將狀態和動作拼接作為輸入
        x = torch.cat([state, action], dim=1)
        return self.network(x)
    
# 定义梯度打印函数，这里加上了参数名的打印
def print_grad_stats(grad, name):
    if grad is not None:
        print(f"{name} - 梯度平均值: {grad.mean()}, 梯度標準差: {grad.std()}, 最大梯度: {grad.max()}, 最小梯度: {grad.min()}")

class MADDPG:
    def __init__(self, state_dim, action_dim, n_agents, actor_lr, critic_lr, gamma, pretrained_actors=None):
        if pretrained_actors is None:
            base_actor = Actor(state_dim, action_dim)  # 初始化一個模型
            self.actors = [copy.deepcopy(base_actor) for _ in range(n_agents)]  # 使用深度複製來創建相同的模型
        else:
            self.actors = pretrained_actors  # 使用提供的模型列表
        self.critic = SingleCritic(state_dim, action_dim, n_agents)
        self.actor_optimizers = [optim.Adam(actor.parameters(), lr=actor_lr) for actor in self.actors]
        self.critic_optimizers = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.n_agents = n_agents
        
        # hook
        for actor in self.actors:
            for name, param in actor.named_parameters():
                if param.requires_grad:
                    # 注意：這裡給hook函數加上了名稱參數
                    param.register_hook(lambda grad, name=name: print_grad_stats(grad, name))
        
        #for name, param in self.critic.named_parameters():
         #   if param.requires_grad:
                # 注意：這裡給hook函數加上了名稱參數
          #      param.register_hook(lambda grad, name=name: print_grad_stats(grad, name))

    def update_agents(self, states, actions, next_states, rewards, dones):
        # states, actions, rewards, next_states 為所有智體的資訊，需要在這裡做處理以適應每個智體

        # 更新Critic
        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.FloatTensor(actions)
        next_states_tensor = torch.FloatTensor(next_states)
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(-1)

        # 計算當前Q值
        current_Q_values = self.critic(states_tensor, actions_tensor)

        
        actions_list = []
        for i in range(next_states_tensor.size(0)):
            # 獲取第 i 個狀態
            state = next_states_tensor[i:i+1]  # 使用切片選取單個狀態，保持維度為 [1, state_dim]
            
            # 透過模型獲取動作
            action = self.actors[0](state, 100)

            action = torch.unsqueeze(action, dim=1)
            
            # 將動作加入列表
            actions_list.append(action)
        
        # 將列表中的動作堆疊起來，形成一個形狀為 [batch_size, action_dim] 的張量
        next_actions_tensor = torch.cat(actions_list, dim=0)

        with torch.no_grad():
            # 計算下一個狀態的Q值
            next_Q_values = self.critic(next_states_tensor, next_actions_tensor)
            # 計算目標Q值
            target_Q_values = rewards_tensor + self.gamma * next_Q_values

        # 計算損失並更新 Critic
        critic_loss = nn.MSELoss()(current_Q_values, target_Q_values)
        self.critic_optimizers.zero_grad()
        critic_loss.backward()
        self.critic_optimizers.step()

        # 更新Actor
        for agent_id in range(self.n_agents):
            self.update_actor(states, actions, agent_id, self.critic)

        return critic_loss.item()

    def update_actor(self, states, actions, agent_id, critic):
        # 將 states, actions 轉成 torch.Tensor
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)

        # 計算actor損失函數
        actor_loss = -critic(states, actions).mean()

        # 更新 Actor
        self.actor_optimizers[agent_id].zero_grad()
        actor_loss.backward()
        self.actor_optimizers[agent_id].step()
        #print(f"Loss: {actor_loss}")
        #return actor_loss.item()
