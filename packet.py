from random import randint
from classifierModel import classifierModel
import numpy as np

import random

# 設置隨機種子
random.seed(5)

'''
學長的封包，應該不用改東西
'''
class Packet:
    def __init__(self, arrive_time, alive_time, packet_info, allocated_slice):
        self.arrive_time = arrive_time                  # 在模擬系統中什麼時候到達？
        self.alive_time = alive_time                    # 在模擬系統中這個封包會占用資源多少時間?
        self.complete_time = arrive_time + alive_time   # 封包在模擬系統中何時完成？
        self.packet_info = packet_info                  # 封包資訊
        self.allocated_slice = allocated_slice          # 分配到的切片


    def generate_packets(model, sys_time, max_alive_time, amount, packet_infos, y):
        packets = {}
        x = 0
        for category in np.unique(y):
            category_indices = np.where(y == category)[0]  # 找到屬於該類別的索引
            indexs = np.random.choice(category_indices, amount, replace=False)  # replace=False 確保不會選擇重複的記錄
            for index in indexs:
                arrive_time = randint(0, sys_time)
                alive_time = randint(1, max_alive_time)
                #print(category, arrive_time, alive_time)
                x += 1
                # 從 pandas 中取得特定的 col
                packet_info = packet_infos.iloc[index : index+1]
                #print(packet_info)
                # 判斷封包要給哪個actor執行動作
                allocated_slice = classifierModel.slice_str_to_int(model.predict(packet_info)[0])
                # 產生新的封包
                packet = Packet(arrive_time, alive_time, packet_info, allocated_slice)
                # 在該單位時間已經有新增封包加入了，我們只要 append list 即可
                if arrive_time in packets:
                    packets[arrive_time].append(packet)
                # 在該單位時間還沒有封包，我們加入含有此 packet 的封包
                else:
                    packets[arrive_time] = [packet]
        return packets
    
    
#print("packet_done")