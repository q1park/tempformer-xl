import torch
import torch.nn as nn

from modules.feedforward import FeedForward
from modules.xlattention import XlAttention

class XlLayer(nn.Module):
    def __init__(self, d_model: int, n_head: int, d_head: int, d_inner: int, drop_out: float, drop_att: float):
        super(XlLayer, self).__init__()

        self.attn = XlAttention(d_model=d_model, n_head=n_head, d_head=d_head, drop_out=drop_out, drop_att=drop_att)
        self.ff = FeedForward(d_in=d_model, d_hidden=d_inner, drop=drop_out)
        self.norm = nn.ModuleList([nn.LayerNorm(d_model),nn.LayerNorm(d_model)])

    def forward(self, x, mem: torch.Tensor, p: torch.Tensor, position: nn.Module, mask: nn.Module) -> torch.Tensor:
        x_mem = torch.cat([mem.to(x.device), x], dim=1)

        x = self.norm[0](x + self.attn(x_mem, p, position, mask=mask, qlen=x.size(1)))
        x = self.norm[1](x + self.ff(x))
        return x