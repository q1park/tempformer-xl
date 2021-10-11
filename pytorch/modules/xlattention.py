import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.xlposition import XlPosition
# from xlposition import XlPosition

class XlAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, d_head: int, drop_out: float, drop_att: float=0.):
        super(XlAttention, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head

        self.W_query = nn.Parameter(torch.Tensor(d_model, n_head, d_head))
        self.W_key = nn.Parameter(torch.Tensor(d_model, n_head, d_head))
        self.W_value = nn.Parameter(torch.Tensor(d_model, n_head, d_head))
        self.W_out = nn.Parameter(torch.Tensor(d_model, n_head, d_head))
        self.W_position = nn.Parameter(torch.Tensor(d_model, n_head, d_head))

        self.drop_out = nn.Dropout(p=drop_out)
        self.drop_att = nn.Dropout(p=drop_att)

        self.scale = 1 / (d_head ** 0.5)

    def forward(self, x, p: torch.Tensor, position: nn.Module, mask: nn.Module, qlen: int) -> torch.Tensor:
        q = self.embed(self.W_query, x[:, -qlen:])
        k = self.embed(self.W_key, x)
        v = self.embed(self.W_value, x)
        r = self.embed(self.W_position, p)

        #### compute attention score (bsz x qlen x klen x n_head)
        AC = torch.einsum('bind,bjnd->bijn', q + position.bias['k'], k)
        BD = torch.einsum('bind,jnd->bijn', q + position.bias['r'], r)
        attn_score = mask(self.scale * (AC + self._rel_shift(BD)))
        attn_score = F.softmax(attn_score, dim=2)

        #### compute output
        x = self.score(self.drop_att(attn_score), v)
        x = self.output(self.W_out, x)
        x = self.drop_out(x)

        return x

    def _rel_shift(self, x, zero_triu: bool=False):
        zero_pad = torch.zeros((x.size(0), x.size(1), 1, x.size(3)), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=2)
        x_padded = x_padded.view(x.size(0), x.size(2) + 1, x.size(1), x.size(3))
        x = x_padded[:, 1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(1), x.size(2)))
            x = x * torch.tril(ones, x.size(2) - x.size(1))[None, :, :, None].to(x.device)

        return x

    def embed(self, op: torch.Tensor, x) -> torch.Tensor:
        return torch.einsum('dha, ...ld -> ...lha', op, x)

    def score(self, op: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum('...lmh, ...mha -> ...lha', op, x)

    def output(self, op: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum('...dha, ...lha -> ...ld', op, x)