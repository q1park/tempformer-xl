import torch
import torch.nn as nn
import torch.nn.functional as F

class XlAttention(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0):
        super(XlAttention, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.q_net = nn.Linear(d_model, n_head * d_head, bias=False)
        self.kv_net = nn.Linear(d_model, 2 * n_head * d_head, bias=False)

        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)
        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)

        self.scale = 1 / (d_head ** 0.5)

    def forward(self, w, r, r_w_bias, r_r_bias, qlen, attn_mask=None):
        bsz, klen, rlen  = w.size(0), w.size(1), r.size(1)

        w_head_q = self._forward_Q(w[:, -qlen:])
        w_head_k, w_head_v = self._forward_KV(w)
        r_head_k = self.r_net(r)

        w_head_q = w_head_q.view(bsz, qlen, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head
        w_head_k = w_head_k.view(bsz, klen, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head
        w_head_v = w_head_v.view(bsz, klen, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head
        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)  # qlen x n_head x d_head

        #### compute attention score
        rw_head_q = w_head_q + r_w_bias  # qlen x bsz x n_head x d_head
        AC = torch.einsum('bind,bjnd->bijn', (rw_head_q, w_head_k))  # qlen x klen x bsz x n_head

        rr_head_q = w_head_q + r_r_bias
        BD = torch.einsum('bind,jnd->bijn', (rr_head_q, r_head_k))  # qlen x klen x bsz x n_head

        attn_score = AC + self._rel_shift(BD) # [qlen x klen x bsz x n_head]
        attn_score.mul_(self.scale)

        #### compute attention probability
        attn_score = attn_mask(attn_score)
        attn_prob = F.softmax(attn_score, dim=2) # [qlen x klen x bsz x n_head]
        attn_prob = self.dropatt(attn_prob)

        #### compute attention vector
        attn_vec = torch.einsum('bijn,bjnd->bind', (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        return attn_out

    def _rel_shift(self, x, zero_triu=False):
        zero_pad = torch.zeros((x.size(0), x.size(1), 1, x.size(3)), device=x.device, dtype=x.dtype)

        x_padded = torch.cat([zero_pad, x], dim=2)
        x_padded = x_padded.view(x.size(0), x.size(2) + 1, x.size(1), x.size(3))

        x = x_padded[:, 1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(1), x.size(2)))
            x = x * torch.tril(ones, x.size(2) - x.size(1))[None, :, :, None].to(x.device)

        return x

    def _forward_Q(self, x):
        return self.q_net(x)

    def _forward_KV(self, x):
        w_head_kv = self.kv_net(x)
        return torch.chunk(w_head_kv, 2, dim=-1)