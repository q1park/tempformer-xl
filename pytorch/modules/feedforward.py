import torch.nn.functional as F
from torch import nn

class FeedForward(nn.Module):
    "Generic Feedforward network."
    def __init__(self, d_in: int, d_hidden: int, drop: float):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(d_in, d_hidden)
        self.w_2 = nn.Linear(d_hidden, d_in)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = F.relu(self.w_1(x))
        x = self.drop(x)
        x = self.w_2(x)
        x = self.drop(x)
        return x