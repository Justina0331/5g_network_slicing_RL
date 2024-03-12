import torch
import numpy as np
from random import randint
from tqdm import tqdm
import pandas as pd  # 假設使用 pandas 來處理 packet_infos 和 real_slices

# 將 csv 中的 slice 欄位轉換為整數
def slice_str_to_int(string):
    slices_mapping = { 'slice_A': 0, 'slice_B': 1, 'slice_C': 2, 'slice_1': 0, 'slice_2': 1, 'slice_3': 2 }
    return slices_mapping[string]

class Packet:
    def __init__(self, arrive_time, alive_time, packet_info):
        self.arrive_time = arrive_time  # 在模擬系統中什麼時候到達？
        self.alive_time = alive_time    # 在模擬系統中這個封包會占用資源多少時間?
        self.packet_info = packet_info  # 封包資訊
        #self.real_slice = real_slice    # 實際應該分配到的切片(RL用Reward來代替評估結果)

    def generate_packets(sys_time, max_alive_time, amount, packet_infos):
        packets = {}
        # 封包資訊有多長?
        packet_infos_len = len(packet_infos)

        for _ in range(amount):
            arrive_time = randint(0, sys_time)
            alive_time = randint(1, max_alive_time)

            # 隨機從 packet_info 中挑一個
            index = randint(0, packet_infos_len)
            # 從 pandas 中取得特定的 col
            packet_info = packet_infos.iloc[index : index+1]
            # 有時候會發生 real_slices index out of bounds...
            # 8:2 可能切出來的東西不對等?
            # real_slice = slice_str_to_int(real_slices.iloc[index])
            # 產生新的封包
            packet = Packet(arrive_time, alive_time, packet_info)
            # 在該單位時間已經有新增封包加入了，我們只要 append list 即可
            if arrive_time in packets:
                packets[arrive_time].append(packet)
            # 在該單位時間還沒有封包，我們加入含有此 packet 的封包
            else:
                packets[arrive_time] = [packet]

        return packets

class Simulator:
    def __init__(self, slices_resource, packet_infos):
        self.slices_resource = slices_resource
        self.packet_infos = packet_infos
        self.ready_packets = {}
        self.running_packets = {}
        self.current_time = 0

    def reset(self):
        self.ready_packets = {}
        self.running_packets = {}
        self.current_time = 0
        # 返回初始狀態
        return self.get_state()

    def step(self, action):
        # 根據動作處理封包，更新環境狀態
        # 這裡的動作處理邏輯需要根據你的具體需求來實現

        # 更新當前時間
        self.current_time += 1

        # 檢查並更新 ready_packets 和 running_packets
        # 這需要根據你的業務邏輯來編寫代碼

        # 計算獎勵
        reward = self.calculate_reward()
        
        # 檢查是否結束
        done = self.check_done()

        # 返回新的狀態、獎勵和是否結束
        return self.get_state(), reward, done, {}
    
    def simulate(self, sys_time=100, max_alive_time=10, sample_interval=10, amount=10):
        # 隨機從 x_test, y_test 中產生 `amount` 個模擬封包，型別為 dict(list(Packet))
        self.ready_packets = Packet.generate_packets(sys_time, max_alive_time, amount, self.packet_infos, self.real_slices)

        # 開始模擬
        for t in tqdm(range(sys_time)):
            # 哪些封包在第 t 個單位時間完成了需要釋放資源?
            self.__release_resource(t)
            # 哪些封包在第 t 個單位時間完成了需要分配資源?
            self.__allocate_resource(t)

            # 我們該取樣系統資源了嗎?
            if t % sample_interval == 0:
                self.__sample(t)
            
            # Comment here when using FCFS
            self.__load_balance()

    def get_state(self):
        # 根據當前環境情況返回狀態
        state = torch.tensor(self.packet_infos.values, dtype=torch.float32)
        return state  # 範例：返回一個空的numpy數組，實際中應返回有效的狀態

    def calculate_reward(self):
        # 根據當前環境情況計算獎勵
        return 0  # 範例：總是返回0，實際中應根據業務邏輯返回有效的獎勵值

    def check_done(self):
        # 根據當前環境情況判斷是否結束
        return False  # 範例：總是返回False，實際中應根據業務邏輯判斷是否結束



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

print("environment_done")
