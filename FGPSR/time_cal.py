import numpy as np
import torch
from torch.backends import cudnn
import time

class TimeRecorder:
    def __init__(self, benchmark):
        if benchmark:
            cudnn.benchmark = True

        # 设置用于测量时间的 cuda Event, 这是PyTorch 官方推荐的接口,理论上应该最靠谱
        self.starter, self.ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        # 初始化一个时间容器
        # self.timings = np.zeros((self.repetitions, 1))
        self.starter_cpu = 0.0
        self.ender_cpu = 0.0

        self.timings_gpu = list()
        self.timings_cpu = list()
        self.timings = list()

        self.avg = 0.0
        self.number = 0


        self.t1 = 0.0
        self.t2 = 0.0
        self.t3 = 0.0
        self.t4 = 0.0

    def start(self):
        self.starter.record()

    def end(self):
        self.ender.record()
        torch.cuda.synchronize()
        curr_time = self.starter.elapsed_time(self.ender)  # 从 starter 到 ender 之间用时,单位为毫秒
        self.timings.append(curr_time)
        self.number = self.number + 1

    def end_t(self):
        self.ender.record()
        torch.cuda.synchronize()
        curr_time = self.starter.elapsed_time(self.ender)  # 从 starter 到 ender 之间用时,单位为毫秒
        self.timings.append(curr_time)

    def start_cpu(self):
        self.starter_cpu = time.perf_counter()

    def end_cpu_t(self):
        self.ender_cpu = time.perf_counter()
        elapsed = self.ender_cpu - self.starter_cpu
        self.timings.append(elapsed*1000)


    def count(self):
        # temp
        self.t4 += self.timings[-1]
        self.t3 += self.timings[-2]
        self.t2 += self.timings[-3]
        self.t1 += self.timings[-4]
        # end temp


        self.number = self.number + 1

    def avg_time_sci(self):
        # avg = (self.timings.sum() - self.timings.max() - self.timings.min()) / (self.number - 2)
        avg = (sum(self.timings) - max(self.timings)) / (self.number - 1)
        print(self.number)
        return avg

    def avg_time(self):
        # avg = (self.timings.sum() - self.timings.max() - self.timings.min()) / (self.number - 2)
        avg = sum(self.timings) / self.number
        print(self.number)

        # temp
        avg1 = self.t1 / self.number
        avg2 = self.t2 / self.number
        avg3 = self.t3 / self.number
        avg4 = self.t4 / self.number
        print('t1: %g,  t2 : %g, t3 : %g, t4 : %g' % (avg1, avg2, avg3, avg4))

        # end temp

        return avg

    def avg_time_pre(self):
        # avg = (self.timings.sum() - self.timings.max() - self.timings.min()) / (self.number - 2)
        avg = (sum(self.timings) - max(self.timings)) / (self.number - 1)
        print(self.number)
        return avg



