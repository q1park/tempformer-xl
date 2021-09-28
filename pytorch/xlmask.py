import torch
from torch import nn


class XlMask(nn.Module):
    """
    Creates, registers, and applies the attention score mask
    """

    def __init__(self, tgt_len: int, mem_len: int, same_length: bool):
        super(XlMask, self).__init__()
        self.tgt_len = tgt_len
        self.klen = tgt_len + mem_len
        self.same_length = same_length

        self.attn_mask = self.make_mask()
        # self.register_buffer('attn_mask', attn_mask)

    def forward(self, x):
        # x.shape = (l_q, l_k, 1, 1)
        mask = self.attn_mask[-x.size(0):, -x.size(1):, None, None]
        # x.masked_fill_(mask.to(x.device), -float('inf'))
        x.masked_fill_(mask.to(x.device), -torch.finfo(x.dtype).max)
        return x

    def make_mask(self):
        # causal_mask = torch.ones(self.tgt_len, self.klen).triu_(self.klen - self.tgt_len + 1).byte()
        causal_mask = torch.ones(self.tgt_len, self.klen).triu_(self.klen - self.tgt_len + 1).bool()

        if self.same_length:
            # causal_mask = causal_mask + torch.ones(self.tgt_len, self.klen).tril_(0).byte()
            causal_mask = causal_mask + torch.ones(self.tgt_len, self.klen).tril_(0).bool()
        return causal_mask