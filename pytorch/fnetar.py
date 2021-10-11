import torch
import torch.nn as nn

from modules.xlmask import XlMask
from modules.xlmemory import XlMemory
from modules.xlposition import XlPosition
from xllayer import XlLayer
from fnetarlayer import FnetarLayer

class Fnetar(nn.Module):
    def __init__(
            self,
            n_layer: int,
            d_model: int,
            n_head: int,
            d_head: int,
            d_inner: int,
            drop_out: float,
            drop_att: float,
            tgt_len: int=None,
            mem_len: int=None,
            same_length: bool=False,
            clamp_len: int=-1
    ):
        super(Fnetar, self).__init__()

        self.position = XlPosition(d_model=d_model, n_head=n_head, d_head=d_head, clamp_len=clamp_len)
        self.layers = nn.ModuleList([
            XlLayer(
                d_model=d_model, n_head=n_head, d_head=d_head, d_inner=d_inner,
                drop_out=drop_out, drop_att=drop_att
            )
            for _ in range(n_layer // 2)
        ] + [
            FnetarLayer(
                d_model=d_model, d_inner=d_inner, drop_out=drop_out,
                tgt_len=tgt_len, mem_len=mem_len, same_length=same_length
            )
            for _ in range(n_layer // 2)
        ])
        self.attn_mask = XlMask(tgt_len=tgt_len, mem_len=mem_len, same_length=same_length)
        self.drop_out = nn.Dropout(drop_out)

    def forward(self, x, memory: XlMemory) -> (torch.Tensor, XlMemory):
        bsz, qlen, mlen = x.size(0), x.size(1), memory.size(0)
        klen = mlen + qlen

        p = self.position.wave_grid(klen=klen, device=x.device, dtype=x.dtype)
        p = self.drop_out(p)

        hids = [x]
        first_fourier = True

        for i, layer in enumerate(self.layers):
            if isinstance(layer, XlLayer):
                x = layer(x=x, mem=memory[i], p=p, position=self.position, mask=self.attn_mask)
            elif isinstance(layer, FnetarLayer):
                if first_fourier:
                    add_position = True
                    first_fourier = False
                else:
                    add_position = False
                x = layer(x=x, mem=memory[i], p=p, add_position=add_position)
            hids.append(x)

        memory.update_memory(hids, memory, mlen, qlen)
        x = self.drop_out(x)

        return x, memory