import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FnetarAttention(nn.Module):
    def __init__(self, tgt_len: int, mem_len: int, dropout=0.1, same_length=True):
        super(FnetarAttention, self).__init__()
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