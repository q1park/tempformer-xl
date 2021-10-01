import torch
import torch.nn as nn

class XlInitializer:
    def __init__(self, init: str='normal', init_range: float=0.1, init_std: float=0.02):
        self.init = init
        self.init_range = init_range
        self.init_std = init_std

    def __call__(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                self.init_weight(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                self.init_bias(m.bias)
        elif classname.find('XlAttention') != -1:
            if hasattr(m, 'W_query'):
                self.init_weight(m.W_query)
            if hasattr(m, 'W_key'):
                self.init_weight(m.W_key)
            if hasattr(m, 'W_value'):
                self.init_weight(m.W_value)
            if hasattr(m, 'W_out'):
                self.init_weight(m.W_out)
            if hasattr(m, 'W_position'):
                self.init_weight(m.W_position)
        elif classname.find('Embedding') != -1:
            if hasattr(m, 'weight'):
                self.init_weight(m.weight)
        elif classname.find('AdaptiveLogSoftmax') != -1:
            if hasattr(m, 'cluster') and m.cluster is not None:
                self.init_weight(m.cluster.weight)
                self.init_bias(m.cluster.bias)
        elif classname.find('LayerNorm') != -1:
            if hasattr(m, 'weight'):
                nn.init.normal_(m.weight, 1.0, self.init_std)
            if hasattr(m, 'bias') and m.bias is not None:
                self.init_bias(m.bias)
        elif classname.find('XlPosition') != -1:
            if hasattr(m.bias, 'k'):
                self.init_weight(m.bias['k'])
            if hasattr(m.bias, 'r'):
                self.init_weight(m.bias['r'])

    def init_weight(self, weight):
        if self.init == 'uniform':
            nn.init.uniform_(weight, -self.init_range, self.init_range)
        elif self.init == 'normal':
            nn.init.normal_(weight, 0.0, self.init_std)

    def init_bias(self, bias):
        nn.init.constant_(bias, 0.0)