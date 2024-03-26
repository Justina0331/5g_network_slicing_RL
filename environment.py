import torch
from packet import Packet
import numpy as np

class Simulator:
    def __init__(self, initial_slices_resource, packet_infos):
        self.initial_slices_resource = initial_slices_resource  # 切片資源初始資源量
        self.GSM_slices_resource = [0, 0, 0]            # GSM給予額外切片資源量
        #self.slices_resource = initial_slices_resource          # 目前切片資源總量
        self.packet_infos = packet_infos                # 封包資訊
        self.total_sent_packets = 0                     # 已發送的封包(包括被拒絕的)
        self.rejected_packets = [0, 0, 0]                       # 被拒絕的封包
        self.allocated_resources = [0, 0, 0]            # 目前被分配正在使用的切片資源量
        self.ready_packets = {}                         # 等待發送的封包
        self.running_packets = {}                       # 正在執行的封包
        self.current_time = 0                           # 當前時間(這東西目前計算是錯的，而且也沒有用到)
        self.re_reward = 0                              # 上一次的獎勵



    def reset(self):
        self.ready_packets = {}
        self.running_packets = {}
        self.current_time = 0

    def step(self, action, packet):
        # 根據動作處理封包，更新環境狀態
        self.total_sent_packets += 1

        '''
        # action[0]決定是否接受封包
        if action[0] == 1 and self.allocated_resources[packet.allocated_slice] < self.slices_resource[packet.allocated_slice]:
            self.allocated_resources[packet.allocated_slice] += 1
            # 分配資源成功，將 packet 放入到處理中的 list 中
            if packet.complete_time in self.running_packets:
                self.running_packets[packet.complete_time].append(packet)
            else:
                self.running_packets[packet.complete_time] = [packet]
        else:
            self.rejected_packets += 1
        
        # action[1]決定是否向GSM要求資源
        if action[1] == 2:      # 向GSM要求資源
            self.GSM_slices_resource[packet.allocated_slice] += 1
            self.slices_resource[packet.allocated_slice] += 1
        elif action[1] == 1:    # 釋放多餘資源給GSM
            if self.slices_resource[packet.allocated_slice] > self.initial_slices_resource[packet.allocated_slice]:
                self.GSM_slices_resource[packet.allocated_slice] -= 1
                self.slices_resource[packet.allocated_slice] -= 1
        '''

        
        # action決定是否向GSM要求資源
        if action == 2:      # 向GSM要求資源
            self.GSM_slices_resource[packet.allocated_slice] += 1
            #self.slices_resource[packet.allocated_slice] += 1
        elif action == 1:    # 釋放多餘資源給GSM
            if self.GSM_slices_resource[packet.allocated_slice] > 0 and self.initial_slices_resource[packet.allocated_slice] + self.GSM_slices_resource[packet.allocated_slice] - self.allocated_resources[packet.allocated_slice] > 0:
                self.GSM_slices_resource[packet.allocated_slice] -= 1
                #self.slices_resource[packet.allocated_slice] -= 1

        self.send_packet(packet)

    
        # 更新當前時間
        self.current_time += 1

        # 計算獎勵
        reward = self.calculate_reward()
        
        # 檢查是否結束
        done = self.check_done()

        # 返回新的狀態、獎勵和是否結束
        return self.get_state(packet.allocated_slice), reward, done, {}
    
    def send_packet(self, packet):
        # 目前資源量可是否接受封包
        if self.allocated_resources[packet.allocated_slice] < self.initial_slices_resource[packet.allocated_slice] + self.GSM_slices_resource[packet.allocated_slice]:
            self.allocated_resources[packet.allocated_slice] += 1
            # 分配資源成功，將 packet 放入到處理中的 list 中
            if packet.complete_time in self.running_packets:
                self.running_packets[packet.complete_time].append(packet)
            else:
                self.running_packets[packet.complete_time] = [packet]
        else:
            self.rejected_packets[packet.allocated_slice] += 1
        
    '''
    學長生成模擬封包的方法
    '''
    def simulate(self, model, sys_time, max_alive_time, amount, y):
        # 隨機從 x_test, y_test 中產生 `amount` 個模擬封包，型別為 dict(list(Packet))
        ready_packets = Packet.generate_packets(model, sys_time, max_alive_time, amount, self.packet_infos, y)
        self.ready_packets = ready_packets
        return ready_packets
        
    
    '''
    性能指標，可看看有沒有其他指標可使用，後續可拿來作圖
    '''
    def performance_metrics(self):
        # 封包拒絕率
        if self.total_sent_packets == 0:
            packet_rejection_rate = 0
        else:

            packet_rejection_rate = sum(self.rejected_packets) / self.total_sent_packets

        # 額外資源使用率
        GSM_usage = [0, 0, 0]
        for i in range(3):
            #print( self.allocated_resources[i], self.initial_slices_resource[i], self.GSM_slices_resource[i])
            # 沒有GSM資源 0
            if self.GSM_slices_resource[i] == 0:
                GSM_usage[i] = 0
            # 有GSM資源且有使用GSM資源 0 ~ 1 不包含0
            elif self.allocated_resources[i] > self.initial_slices_resource[i]:
                in_GSM = self.allocated_resources[i] - self.initial_slices_resource[i]
                GSM_usage[i] = in_GSM / self.GSM_slices_resource[i]
            # 有GSM資源但沒使用 -1 ~ 0 不包含0
            else:
                # 計算要求但未使用的GSM資源比例
                requested_unutilized = self.initial_slices_resource[i] - self.allocated_resources[i]
                # 根據要求但未使用的比例計算懲罰，這裡要求的越多懲罰越重
                GSM_usage[i] = -requested_unutilized / self.initial_slices_resource[i]
        #print(GSM_usage)
        #print(GSM_usage)
        GSM_resource_usage = sum(GSM_usage)
        #print(GSM_resource_usage, packet_rejection_rate)

        return GSM_resource_usage, packet_rejection_rate

    def get_state(self, slice):
        # 根據當前環境情況返回狀態
        
        #slices_resource_tensor = torch.tensor(self.slices_resource[slice], dtype=torch.float32).unsqueeze(0)
        GSM_slices_resource_tensor = torch.tensor(self.GSM_slices_resource[slice], dtype=torch.float32).unsqueeze(0)
        #total_sent_packets_tensor = torch.tensor(self.total_sent_packets).unsqueeze(0)
        #rejected_packets_tensor = torch.tensor(self.rejected_packets).unsqueeze(0)
        rejected_packets_tensor = torch.tensor(self.rejected_packets[slice], dtype=torch.float32).unsqueeze(0)
        #total_allocated_resources_tensor = torch.tensor(self.allocated_resources[slice], dtype=torch.float32).unsqueeze(0)
        idle_slices_resource_tensor = torch.tensor(self.initial_slices_resource[slice] + self.GSM_slices_resource[slice] - self.allocated_resources[slice], dtype=torch.float32).unsqueeze(0)
        idle_initial_slices_resource_tensor = torch.tensor(0 if self.initial_slices_resource[slice] < self.allocated_resources[slice] else self.initial_slices_resource[slice] - self.allocated_resources[slice], dtype=torch.float32).unsqueeze(0)
        idle_GSM_slices_resource_tensor = torch.tensor(self.GSM_slices_resource[slice] if self.initial_slices_resource[slice] > self.allocated_resources[slice] else self.initial_slices_resource[slice] + self.GSM_slices_resource[slice] - self.allocated_resources[slice], dtype=torch.float32).unsqueeze(0)
        #send_tensor = torch.tensor(send).unsqueeze(0)
        '''
        slices_resource_tensor = torch.tensor(self.slices_resource, dtype=torch.float32)
        GSM_slices_resource_tensor = torch.tensor(self.GSM_slices_resource[slice], dtype=torch.float32).unsqueeze(0)
        rejected_packets_tensor = torch.tensor(self.rejected_packets).unsqueeze(0)
        total_allocated_resources_tensor = torch.tensor(self.allocated_resources, dtype=torch.float32)
        '''


        #ready_packets_tensor = torch.tensor(self.ready_packets, dtype=torch.float32)
        #running_packets_tensor = torch.tensor(self.running_packets, dtype=torch.float32)
        # 選擇的切片的資源總數, 向GSM要求的資源量, 封包拒絕數量, 選擇的切片正在使用的資源量
        #state = torch.cat((slices_resource_tensor, GSM_slices_resource_tensor, rejected_packets_tensor,
        #                   total_allocated_resources_tensor), dim = 0)
        state = torch.cat((idle_initial_slices_resource_tensor, idle_GSM_slices_resource_tensor), dim = 0)
        #state = torch.cat((idle_slices_resource_tensor, rejected_packets_tensor), dim = 0)
        return state

    
    # 可更改，也可看看有沒有其他指標可作為獎勵
    
    def calculate_reward(self):
        # 根據當前環境情況計算獎勵
        GSM_resource_usage, packet_rejection_rate = self.performance_metrics()
        #reward = 0.2 * GSM_resource_usage + 1 * (1 - packet_rejection_rate)
        reward = (0.3 * ((GSM_resource_usage + 3) / 6)) + (0.7 * (1 - packet_rejection_rate))
        #reward = GSM_resource_usage - packet_rejection_rate * 0.5
        #print(0.6 * (GSM_resource_usage + 3) / 6, 0.4 * (1 - packet_rejection_rate), reward)
        return reward
    
    
    '''
    def calculate_reward(self):
        GSM_resource_usage, packet_rejection_rate = self.performance_metrics()
        reward = 0.2 * GSM_resource_usage + 0.4 * (1 - packet_rejection_rate)
        
        # 更新指數移動平均
        smoothing_factor = 0.1  # 指數移動平均的平滑係數
        self.re_reward = smoothing_factor * reward + (1 - smoothing_factor) * self.re_reward

        return self.re_reward
    '''

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
