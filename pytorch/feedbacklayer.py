import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch import einsum
from einops import rearrange
from modules.feedbackutils import exists, safe_cat


# positional embedding

class RelativePositionBias(nn.Module):
    def __init__(
            self,
            causal=False,
            num_buckets=32,
            max_distance=128,
            heads=8
    ):
        super().__init__()
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, causal=True, num_buckets=32, max_distance=128):
        ret = 0
        n = -relative_position
        if not causal:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
                torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, qk_dots):
        i, j, device = *qk_dots.shape[-2:], qk_dots.device
        q_pos = torch.arange(i, dtype=torch.long, device=device)
        k_pos = torch.arange(j, dtype=torch.long, device=device)
        rel_pos = k_pos[None, :] - q_pos[:, None]
        rp_bucket = self._relative_position_bucket(rel_pos, causal=self.causal, num_buckets=self.num_buckets,
                                                   max_distance=self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        bias = rearrange(values, 'i j h -> () h i j')
        return bias


# helper classes

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


class SkipIf(nn.Module):
    def __init__(self, cond, fn):
        super().__init__()
        self.cond = cond
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        if self.cond(x, *args, **kwargs):
            return x
        return self.fn(x, *args, **kwargs)


# feedforward

class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.gelu(gate) * x


class FeedForward(nn.Module):
    def __init__(
            self,
            *,
            dim,
            mult=4,
            dropout=0.
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)


# attention

class Attention(nn.Module):
    def __init__(
            self,
            *,
            dim,
            heads=8,
            dim_head=64,
            dropout=0.
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.scale = dim_head ** -0.5

        inner_dim = dim_head * heads
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, pos_emb=None):
        h, n, device = self.heads, x.shape[1], x.device

        self_attend = n > 1  # only self attend if going at greater than 1 token at a time

        q = self.to_q(x) * self.scale

        k, v = memory if exists(memory) else (None, None)

        if self_attend:
            self_k, self_v = self.to_kv(x).chunk(2, dim=-1)
            k = safe_cat(k, self_k, dim=1)
            v = safe_cat(v, self_v, dim=1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        i, j = sim.shape[-2:]

        if exists(pos_emb):
            sim = sim + pos_emb(sim)

        if self_attend:
            causal_mask = torch.ones(i, j, device=device).triu_(j - i + 1).bool()
            causal_mask = rearrange(causal_mask, 'i j -> () () i j')
            mask_value = -torch.finfo(q.dtype).max
            sim.masked_fill_(causal_mask, mask_value)

        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class FeedbackLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, dropatt, pre_lnorm=None):
        super(FeedbackLayer, self).__init__()
        self.d_model = d_model
        self.dec_attn = Attention(dim=d_model, heads=n_head, dim_head=d_head, dropout=dropatt)
        self.pos_ff = FeedForward(dim=d_model, dropout=dropout)

    def init_module(self, shared_kv_proj, seq_len):
        shared_kv_proj = shared_kv_proj if exists(shared_kv_proj) else self.dec_attn.to_kv
        self.dec_attn.to_kv = shared_kv_proj

        self.dec_attn, self.pos_ff = map(
            lambda fn: Residual(PreNorm(self.d_model, fn)),
            (self.dec_attn, self.pos_ff)
        )

        if seq_len == 1:
            memory_is_empty = lambda *args, **kwargs: not exists(kwargs['memory'])
            attn = SkipIf(memory_is_empty, self.dec_attn)

        return shared_kv_proj

    def forward(self, dec_inp, r, r_w_bias=None, r_r_bias=None, mems=None, dec_attn_mask=None):
        output = self.dec_attn(dec_inp, memory=mems, pos_emb=r)
        #         (dec_inp, r, r_w_bias, r_r_bias, attn_mask=dec_attn_mask, mems=mems)
        output = self.pos_ff(output)

        return output