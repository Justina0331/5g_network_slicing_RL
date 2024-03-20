import torch
from packet import Packet

class Simulator:
    def __init__(self, slices_resource, packet_infos):
        self.slices_resource = slices_resource  # 目前切片資源總量
        self.packet_infos = packet_infos        # 封包資訊
        self.total_sent_packets = 0             # 已發送的封包(包括被拒絕的)
        self.rejected_packets = 0               # 被拒絕的封包
        self.allocated_resources = [0, 0, 0]    # 目前被分配正在使用的切片資源量
        self.ready_packets = {}                 # 等待發送的封包
        self.running_packets = {}               # 正在執行的封包
        self.current_time = 0                   # 當前時間(這東西目前計算是錯的，而且也沒有用到)

    def reset(self):
        self.ready_packets = {}
        self.running_packets = {}
        self.current_time = 0

    def step(self, action, packet):
        # 根據動作處理封包，更新環境狀態
        self.total_sent_packets += 1

        # action[0]決定是否接受封包
        if action[0] > -1 and self.allocated_resources[packet.allocated_slice] < self.slices_resource[packet.allocated_slice]:
            self.allocated_resources[packet.allocated_slice] += 1
            # 分配資源成功，將 packet 放入到處理中的 list 中
            if packet.complete_time in self.running_packets:
                self.running_packets[packet.complete_time].append(packet)
            else:
                self.running_packets[packet.complete_time] = [packet]
        else:
            self.rejected_packets += 1

        # action[1]決定是否向GSM要求資源
        if action[1] < -0.6:
            self.slices_resource[packet.allocated_slice] += 1
        elif action[1] > 0.6:
            self.slices_resource[packet.allocated_slice] -= 1
        
        # 更新當前時間
        self.current_time += 1

        # 計算獎勵
        reward = self.calculate_reward()
        
        # 檢查是否結束
        done = self.check_done()

        # 返回新的狀態、獎勵和是否結束
        return self.get_state(), reward, done, {}
    
    '''
    學長生成模擬封包的方法
    '''
    def simulate(self, model, sys_time=100, max_alive_time=10, amount=10):
        # 隨機從 x_test, y_test 中產生 `amount` 個模擬封包，型別為 dict(list(Packet))
        self.ready_packets = Packet.generate_packets(model, sys_time, max_alive_time, amount, self.packet_infos)
    
    '''
    性能指標，可看看有沒有其他指標可使用，後續可拿來作圖
    '''
    def performance_metrics(self):
        # 封包拒絕率
        if self.total_sent_packets == 0:
            packet_rejection_rate = 0
        else:
            packet_rejection_rate = self.rejected_packets / self.total_sent_packets

        # 資源使用率
        if sum(self.slices_resource) == 0:
            resource_usage = 0
        else:
            resource_usage = sum(self.allocated_resources) / sum(self.slices_resource)

        return resource_usage, packet_rejection_rate


    def get_state(self):
        # 根據當前環境情況返回狀態
        slices_resource_tensor = torch.tensor(self.slices_resource, dtype=torch.float32)
        total_sent_packets_tensor = torch.tensor(self.total_sent_packets).unsqueeze(0)
        rejected_packets_tensor = torch.tensor(self.rejected_packets).unsqueeze(0)
        total_allocated_resources_tensor = torch.tensor(self.allocated_resources, dtype=torch.float32)

        #ready_packets_tensor = torch.tensor(self.ready_packets, dtype=torch.float32)
        #running_packets_tensor = torch.tensor(self.running_packets, dtype=torch.float32)
        state = torch.cat((slices_resource_tensor, total_sent_packets_tensor, rejected_packets_tensor, 
                           total_allocated_resources_tensor), dim = 0)
        return state

    '''
    可更改，也可看看有沒有其他指標可作為獎勵
    '''
    def calculate_reward(self):
        # 根據當前環境情況計算獎勵
        resource_usage, packet_rejection_rate = self.performance_metrics()
        reward = 0.8 * resource_usage + 0.2 * (1 - packet_rejection_rate)
        return reward

    def check_done(self):
        # 根據當前環境情況判斷是否結束
        # 還沒設計，也不一定用到
        return False



'''
# 範例：使用 pandas 創建 packet_infos 和 real_slices
packet_infos = pd.DataFrame({'info': ['info1', 'info2', 'info3']})
real_slices = pd.Series(['slice_1', 'slice_2', 'slice_3'])

# 初始化模擬器
env = Simulator(10, packet_infos, real_slices)

# 使用 reset() 和 step(action) 方法
initial_state = env.reset()
#new_state, reward, done, _ = env.step(action)
'''

#print("environment_done")
