import torch
from packet import Packet

# 將 csv 中的 slice 欄位轉換為整數
def slice_str_to_int(string):
    slices_mapping = { 'slice_A': 0, 'slice_B': 1, 'slice_C': 2, 'slice_1': 0, 'slice_2': 1, 'slice_3': 2 }
    return slices_mapping[string]

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
        self.ready_packets = Packet.generate_packets(sys_time, max_alive_time, amount, self.packet_infos)

        return self.ready_packets

    def get_state(self):
        # 根據當前環境情況返回狀態
        slices_resource_tensor = torch.tensor(self.slices_resource, dtype=torch.float32)

        #ready_packets_tensor = torch.tensor(self.ready_packets, dtype=torch.float32)
        #running_packets_tensor = torch.tensor(self.running_packets, dtype=torch.float32)
        #state = torch.cat((slices_resource_tensor, ready_packets_tensor, running_packets_tensor), dim = 0)
        state = slices_resource_tensor
        return state

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
