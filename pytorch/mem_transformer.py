import torch
import torch.nn as nn

from adaptiveinput import AdaptiveInput
from adaptivelogsoftmax import AdaptiveLogSoftmax
from feedforward import FeedForward
from xlattention import XlAttention
from xlmask import XlMask
from xlmemory import XlMemory
from xlposition import XlPosition

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Fourier(nn.Module):
    def __init__(self, tgt_len: int, mem_len: int, dropout=0.1, same_length=True):
        super(Fourier, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.same_length = same_length

        self.ft_matrix = self.init_ft_matrix(qlen=tgt_len, klen=tgt_len + mem_len)

    def init_ft_matrix(self, qlen, klen):
        flen = klen - qlen + 1
        ft_matrix = torch.outer(torch.arange(0, qlen), torch.arange(0, flen))
        ft_matrix = torch.exp(2 * np.pi * 1j * ft_matrix / flen) / np.sqrt(flen)
        ft_matrix = F.pad(ft_matrix, (klen - flen, 0))

        for i in range(qlen):
            ft_matrix[i] = torch.roll(ft_matrix[i], -(qlen - i - 1))

        ft_matrix = torch.tril(ft_matrix, (klen - qlen))

        if self.same_length:
            ft_matrix = torch.triu(ft_matrix, 0)
        return ft_matrix

    def forward(self, x, p, qlen: int, add_position: bool):
        if add_position:
            x = x + p

        x = self.dropout(x)

        ft_matrix = self.ft_matrix[-qlen:, -x.size(1):].to(x.device)
        # ft_matrix = self.init_ft_matrix(qlen=qlen, klen=x.size(1)).to(x.device)

        x = torch.einsum('lm,bmd->bld', ft_matrix, x.type_as(ft_matrix)) / np.sqrt(x.size(1))
        # x = torch.fft.fft(x, dim=-1).real
        x = x.real
        return x

class FnetarLayer(nn.Module):
    def __init__(self, d_model: int, d_inner: int, drop_out: float, tgt_len: int, mem_len: int, same_length: bool=True):
        super(FnetarLayer, self).__init__()

        self.frr = Fourier(tgt_len=tgt_len, mem_len=mem_len, dropout=drop_out, same_length=same_length)
        self.ff = FeedForward(d_in=d_model, d_hidden=d_inner, drop=drop_out)
        self.norm = nn.ModuleList([nn.LayerNorm(d_model),nn.LayerNorm(d_model)])

    def forward(self, x, mem: torch.Tensor, p: torch.Tensor, add_position: bool) -> torch.Tensor:
        x_mem = torch.cat([mem.to(x.device), x], dim=1)

        x = self.norm[0](x + self.frr(x_mem, p, qlen=x.size(1), add_position=add_position))
        x = self.norm[1](x + self.ff(x))
        return x

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

class Xl(nn.Module):
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
        super(Xl, self).__init__()

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

class MemTransformerLM(nn.Module):
    def __init__(
            self,
            n_token: int,
            n_layer: int,
            n_head: int,
            d_model: int,
            d_head: int,
            d_inner: int,
            drop_out: float,
            drop_att: float,
            tie_weight: bool=True,
            d_embed: int=None,
            div_val: int=1,
            tgt_len: int=None,
            ext_len: int=None,
            mem_len: int=None,
            cutoffs: list=[],
            same_length: bool=False,
            clamp_len: int=-1,
    ):
        super(MemTransformerLM, self).__init__()
        self.n_token = n_token
        self.n_layer = n_layer
        self.n_head = n_head
        self.d_head = d_head

        d_embed = d_model if d_embed is None else d_embed
        self.d_embed = d_embed
        self.d_model = d_model

        self.embedder = AdaptiveInput(d_model=d_model, n_classes=n_token, cutoffs=cutoffs, div_value=div_val)
        self.transformer = Xl(
            n_layer=n_layer, d_model=d_model, n_head=n_head, d_head=d_head, d_inner=d_inner,
            drop_out=drop_out, drop_att=drop_att, tgt_len=tgt_len, mem_len=mem_len,
            same_length=same_length, clamp_len=clamp_len
        )
        self.predictor = AdaptiveLogSoftmax(d_model=d_model, n_classes=n_token, cutoffs=cutoffs, div_value=div_val)
        self.drop_out = nn.Dropout(drop_out)

        if tie_weight:
            self._tie_weights()


    def forward(self, x, y: torch.Tensor, memory: XlMemory) -> (torch.Tensor, XlMemory):
        x = self.embedder(x)
        x = self.drop_out(x)

        x, new_memory = self.transformer(x=x, memory=memory)

        output = self.predictor(x.view(-1, x.size(-1)), y.view(-1))
        loss = -output.output.view(y.size(1), -1)

        return loss, new_memory

    def _tie_weights(self):
        self.embedder.head.weight = self.predictor.head.weight
        for i in range(len(self.embedder.cutoffs) - 1):
            self.embedder.tail[i].weight = self.predictor.tail[i].weight




if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='unit test')

    parser.add_argument('--n_layer', type=int, default=4, help='')
    parser.add_argument('--n_rel_layer', type=int, default=4, help='')
    parser.add_argument('--n_head', type=int, default=2, help='')
    parser.add_argument('--d_head', type=int, default=2, help='')
    parser.add_argument('--d_model', type=int, default=200, help='')
    parser.add_argument('--d_embed', type=int, default=200, help='')
    parser.add_argument('--d_inner', type=int, default=200, help='')
    parser.add_argument('--dropout', type=float, default=0.0, help='')
    parser.add_argument('--cuda', action='store_true', help='')
    parser.add_argument('--seed', type=int, default=1111, help='')
    parser.add_argument('--multi_gpu', action='store_true', help='')

    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda else "cpu")

    B = 4
    tgt_len, mem_len, ext_len = 36, 36, 0
    data_len = tgt_len * 20
    args.n_token = 10000

    from pytorch import data_utils

    data = torch.LongTensor(data_len*B).random_(0, args.n_token).to(device)
    diter = data_utils.LMOrderedIterator(data, B, tgt_len, device=device, ext_len=ext_len)

    cutoffs = [args.n_token // 2]
    tie_projs = [False] + [True] * len(cutoffs)

    for div_val in [1, 2]:
        for d_embed in [200, 100]:
            model = MemTransformerLM(args.n_token, args.n_layer, args.n_head,
                                     args.d_model, args.d_head, args.d_inner, args.dropout,
                                     drop_att=args.dropout, tie_weight=True,
                                     d_embed=d_embed, div_val=div_val,
                                     tie_projs=tie_projs, pre_lnorm=True,
                                     tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
                                     cutoffs=cutoffs).to(device)

            print(sum(p.numel() for p in model.parameters()))

            mems = tuple()
            for idx, (inp, tgt, seqlen) in enumerate(diter):
                print('batch {}'.format(idx))
                out = model(inp, tgt, *mems)
                mems = out[1:]
