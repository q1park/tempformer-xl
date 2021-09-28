import sys
import math
import functools

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('utils')

class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.outer(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:,None,:].expand(-1, bsz, -1)
        else:
            return pos_emb[:,None,:]


class RelMultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0):
        super(RelMultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.q_net = nn.Linear(d_model, n_head * d_head, bias=False)
        self.kv_net = nn.Linear(d_model, 2 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)
        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)

        self.scale = 1 / (d_head ** 0.5)

    def forward(self, w, r, r_w_bias, r_r_bias, qlen, attn_mask=None):
        klen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        w_head_q = self._forward_Q(w[-qlen:])
        w_head_k, w_head_v = self._forward_KV(w)
        r_head_k = self.r_net(r)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head
        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)  # qlen x n_head x d_head

        #### compute attention score
        rw_head_q = w_head_q + r_w_bias  # qlen x bsz x n_head x d_head
        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))  # qlen x klen x bsz x n_head

        rr_head_q = w_head_q + r_r_bias
        BD = torch.einsum('ibnd,jnd->ijbn', (rr_head_q, r_head_k))  # qlen x klen x bsz x n_head
        BD = self._rel_shift(BD)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        #### compute attention probability
        if attn_mask is not None:
            attn_score = attn_mask(attn_score)

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        #### compute attention vector
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        return attn_out

    def _rel_shift(self, x, zero_triu=False):
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]), device=x.device, dtype=x.dtype)

        x_padded = torch.cat([zero_pad, x], dim=1)
        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])

        x = x_padded[1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:,:,None,None]

        return x

    def _forward_Q(self, x):
        return self.q_net(x)

    def _forward_KV(self, x):
        w_head_kv = self.kv_net(x)
        return torch.chunk(w_head_kv, 2, dim=-1)



from adaptiveinput import AdaptiveInput
from adaptivelogsoftmax import AdaptiveLogSoftmax
from feedforward import FeedForward
from xlmask import XlMask

class RelPartialLearnableDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, dropatt):
        super(RelPartialLearnableDecoderLayer, self).__init__()

        self.attn = RelMultiHeadAttn(n_head, d_model, d_head, dropout, dropatt)
        self.ff = FeedForward(d_model, d_inner, dropout)

        self.norm = nn.ModuleList([
            nn.LayerNorm(d_model),
            nn.LayerNorm(d_model)
        ])

    def forward(self, x, r, r_w_bias, r_r_bias, dec_attn_mask=None, mems=None):
        if mems is not None:
            x_mem = torch.cat([mems.to(x.device), x], dim=0)

        x = self.norm[0](
            x + self.attn(x_mem, r, r_w_bias, r_r_bias, qlen=x.size(0), attn_mask=dec_attn_mask))
        x = self.norm[1](
            x + self.ff(x))
        return x

from xlmemory import XlMemory
from xlmemories import XlMemories

class MemTransformerLM(nn.Module):
    def __init__(
            self, n_token, n_layer, n_head, d_model, d_head, d_inner, dropout, dropatt,
            tie_weight=True, d_embed=None, div_val=1, pre_lnorm=False,
            tgt_len=None, ext_len=None, mem_len=None, cutoffs=[], same_length=False, clamp_len=-1
    ):
        super(MemTransformerLM, self).__init__()
        self.n_token = n_token
        self.n_layer = n_layer

        d_embed = d_model if d_embed is None else d_embed
        self.d_embed = d_embed
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head

        self.word_emb = AdaptiveInput(d_model=d_model, n_classes=n_token, cutoffs=cutoffs, div_value=div_val)

        self.layers = nn.ModuleList()

        for i in range(n_layer):
            self.layers.append(
                RelPartialLearnableDecoderLayer(n_head, d_model, d_head, d_inner, dropout, dropatt)
            )

        self.mask = XlMask(tgt_len=tgt_len, mem_len=mem_len, same_length=same_length)

        # use adaptive softmax (including standard softmax)
        self.crit = AdaptiveLogSoftmax(d_model=d_model, n_classes=n_token, cutoffs=cutoffs, div_value=div_val)

        self.drop = nn.Dropout(dropout)

        if tie_weight:
            self.word_emb.head.weight = self.crit.head.weight
            for i in range(len(self.word_emb.cutoffs) - 1):
                self.word_emb.tail[i].weight = self.crit.tail[i].weight

        self.same_length = same_length
        self.clamp_len = clamp_len

        self._create_params()

    def _create_params(self):
        self.pos_emb = PositionalEmbedding(self.d_model)
        self.r_w_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
        self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))

    def _forward(self, dec_inp, memory=None):
        qlen, bsz = dec_inp.size()

        word_emb = self.word_emb(dec_inp)

        mlen = memory[0].size(0) if memory is not None else 0
        klen = mlen + qlen

        hids = []
        pos_seq = torch.arange(klen-1, -1, -1.0, device=word_emb.device, dtype=word_emb.dtype)

        if self.clamp_len > 0:
            pos_seq.clamp_(max=self.clamp_len)
        pos_emb = self.pos_emb(pos_seq)

        core_out = self.drop(word_emb)
        pos_emb = self.drop(pos_emb)

        hids.append(core_out)
        for i, layer in enumerate(self.layers):
            core_out = layer(
                core_out, pos_emb, self.r_w_bias, self.r_r_bias,
                dec_attn_mask=self.mask, mems=memory[i]
            )
            hids.append(core_out)


        core_out = self.drop(core_out)

        memory.update_memory(hids, memory, mlen, qlen)

        return core_out, memory

    def forward(self, data, target, memory: XlMemory):
        tgt_len = target.size(0)
        hidden, new_mems = self._forward(data, memory=memory)

        pred_hid = hidden[-tgt_len:]
        output = self.crit(pred_hid.view(-1, pred_hid.size(-1)), target.view(-1))
        loss = -output.output.view(tgt_len, -1)

        return loss, new_mems

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

    import data_utils

    data = torch.LongTensor(data_len*B).random_(0, args.n_token).to(device)
    diter = data_utils.LMOrderedIterator(data, B, tgt_len, device=device, ext_len=ext_len)

    cutoffs = [args.n_token // 2]
    tie_projs = [False] + [True] * len(cutoffs)

    for div_val in [1, 2]:
        for d_embed in [200, 100]:
            model = MemTransformerLM(args.n_token, args.n_layer, args.n_head,
                            args.d_model, args.d_head, args.d_inner, args.dropout,
                            dropatt=args.dropout, tie_weight=True, 
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
