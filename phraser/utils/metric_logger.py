from datetime import timedelta
import time
from collections import defaultdict

import math
import numpy as np
import psutil
import torch
from torch.utils.tensorboard import SummaryWriter

MB = 1024.0 * 1024.0

class MetricLogger(object):
    def __init__(self, tensorboard, log_frequency=-1, memory_frequency=None, name="", delimiter="\t"):
        self.meters = defaultdict(lambda : [])
        self.delimiter = delimiter
        self.tensorboard: SummaryWriter = tensorboard
        self.name = name
        self.log_frequency = log_frequency

        if memory_frequency is None:
            memory_frequency = log_frequency

        self.memory_frequency = memory_frequency
        self.curr_index = 1

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = list(v.cpu().detach().numpy().reshape(-1))



            if isinstance(v, (float, int, np.floating)):
                v = [float(v)]

            if isinstance(v, (np.ndarray, np.generic)):
                v = list(v)

            assert isinstance(v, (list) )
            self.meters[k].extend(v)

    def get(self, k):
        return np.array(self.meters[k]).mean()

    def avg(self):
        for k, v in self.meters.items():
            yield k, np.array(v).mean()
    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            meter = np.array(meter)
            loss_str.append(
                "{}: {}±{}".format(name, str(meter.mean()), str(meter.std()))
            )
        return self.delimiter.join(loss_str)

    def log(self,  reset = True):
        freq = abs(self.log_frequency)

        for k, value in self.avg():
            self.tensorboard.add_scalar(f'{self.name}/{k}', value, self.curr_index//freq )

        self.curr_index += 1

        if reset:
            self.reset()

    def reset(self):
        self.meters = defaultdict(lambda : [])

    def auto_log(self):
        if self.log_frequency != -1:
            if self.curr_index % self.log_frequency == 0:
                self.log()
            else:
                self.curr_index += 1

