from random import randint

class Packet:
    def __init__(self, arrive_time, alive_time, packet_info):
        self.arrive_time = arrive_time  # 在模擬系統中什麼時候到達？
        self.alive_time = alive_time    # 在模擬系統中這個封包會占用資源多少時間?
        self.packet_info = packet_info  # 封包資訊
        self.allocated_slice = -1       # 分配到的切片

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
            # 產生新的封包
            packet = Packet(arrive_time, alive_time, packet_info)
            # 在該單位時間已經有新增封包加入了，我們只要 append list 即可
            if arrive_time in packets:
                packets[arrive_time].append(packet)
            # 在該單位時間還沒有封包，我們加入含有此 packet 的封包
            else:
                packets[arrive_time] = [packet]

        return packets