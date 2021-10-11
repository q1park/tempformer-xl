import torch
import torch.nn as nn

from modules.feedforward import FeedForward
from modules.fnetarattention import FnetarAttention


class FnetarLayer(nn.Module):
    def __init__(self, d_model: int, d_inner: int, drop_out: float, tgt_len: int, mem_len: int, same_length: bool=True):
        super(FnetarLayer, self).__init__()

        self.frr = FnetarAttention(tgt_len=tgt_len, mem_len=mem_len, dropout=drop_out, same_length=same_length)
        self.ff = FeedForward(d_in=d_model, d_hidden=d_inner, drop=drop_out)
        self.norm = nn.ModuleList([nn.LayerNorm(d_model),nn.LayerNorm(d_model)])

    def forward(self, x, mem: torch.Tensor, p: torch.Tensor, add_position: bool) -> torch.Tensor:
        x_mem = torch.cat([mem.to(x.device), x], dim=1)

        x = self.norm[0](x + self.frr(x_mem, p, qlen=x.size(1), add_position=add_position))
        x = self.norm[1](x + self.ff(x))
        return x