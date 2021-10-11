import math
from collections import namedtuple

import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange

# constants

Memory = namedtuple('Memory', ['keys', 'values'])


# helpers

def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def safe_cat(arr, el, dim=1):
    if not exists(arr):
        return el
    return torch.cat((arr, el), dim=dim)


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


class DecoderFbLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, dropatt, pre_lnorm=None):
        super(DecoderFbLayer, self).__init__()
        self.d_model = d_model
        self.dec_attn = Attention(dim=d_model, heads=n_head, dim_head=d_head, dropout=dropatt)
        self.pos_ff = FeedForward(dim=d_model, dropout=dropout)

    def init_module(self, shared_kv_proj, seq_len):
        shared_kv_proj = default(shared_kv_proj, self.dec_attn.to_kv)
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


class DecoderFb(nn.Module):
    def __init__(
            self, n_layer, n_head, d_model, d_head=64,
            d_inner=None, dropout=0., dropatt=0.,
            mem_len=150, seq_len=150, keep_last_hidden=False,
            same_length=False, pre_lnorm=False):
        super(DecoderFb, self).__init__()
        self.n_layer = n_layer
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head
        self.d_inner = d_inner
        self.seq_len = seq_len
        self.mem_len = mem_len
        self.same_length = same_length
        self.pre_lnorm = pre_lnorm

        # main layers

        self.layers = nn.ModuleList([])
        shared_kv_proj = None

        for _ in range(n_layer):
            self.layers.append(DecoderFbLayer(
                n_head=n_head, d_model=d_model, d_head=d_head, d_inner=None,
                dropout=dropout, dropatt=dropatt
            ))
            shared_kv_proj = self.layers[-1].init_module(shared_kv_proj=shared_kv_proj, seq_len=self.seq_len)

        # memory parameters

        self.layer_weight = nn.Parameter(torch.ones(n_layer + 1))
        self.shared_kv_proj = shared_kv_proj
        self.keep_last_hidden = keep_last_hidden

    def get_layer_weight(self):
        layer_weight = self.layer_weight.softmax(dim=-1)
        layer_weight = rearrange(layer_weight, 'd -> d () () ()')
        return layer_weight

    def forward(self, dec_inp, pos_emb, mems, layer_weight, dec_attn_mask=None):
        if exists(mems):
            memory_keys, memory_values = mems

        dec_outp = dec_inp
        # prepare memory for attention, if it exists
        hiddens = [dec_outp]

        memory = None
        if exists(memory_keys):
            memory = (memory_keys, memory_values)

        for layer in self.layers:
            dec_outp = layer(dec_inp=dec_outp, r=pos_emb, mems=memory)
            hiddens.append(dec_outp)

        # calculate new memory key / values and store to FIFO queue

        if self.keep_last_hidden:  # secret option for only keeping last hidden layer, as in paper
            agg_hiddens = hiddens[-1]
        else:
            hiddens = torch.stack(hiddens)
            agg_hiddens = (hiddens * layer_weight).sum(dim=0)

        # pre-calculate memory key / values and store to buffer

        mem_k, mem_v = self.shared_kv_proj(agg_hiddens).chunk(2, dim=-1)
        memory_keys = safe_cat(memory_keys, mem_k, dim=1)
        memory_values = safe_cat(memory_values, mem_v, dim=1)

        # enforce max length on memory buffer

        memory_keys = memory_keys[:, -self.mem_len:]
        memory_values = memory_values[:, -self.mem_len:]

        output = {
            'output': dec_outp,
            'mems': Memory(memory_keys, memory_values),
            'agg_hiddens': agg_hiddens
        }
        return output