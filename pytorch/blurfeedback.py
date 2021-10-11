import torch
import torch.nn as nn

from modules.adaptiveinput import AdaptiveInput
from modules.adaptivelogsoftmax import AdaptiveLogSoftmax
from modules.feedbackmemories import FeedbackMemory
from modules.feedbackutils import exists
from feedbacklayer import RelativePositionBias
from feedback import Feedback

from xlinitializer import XlInitializer as weights_init


class BlurFeedback(nn.Module):
    def __init__(
            self,
            n_token, n_layer, n_head, d_model, d_head, d_inner, drop_out, drop_att,
            tie_weight=True, d_embed=None, div_val=1,
            tgt_len=None, mem_len=None, ext_len=None,
            cutoffs=[], same_length=False, clamp_len=-1
    ):
        super(BlurFeedback, self).__init__()
        self.n_token = n_token
        self.n_layer = n_layer
        self.n_head = n_head
        self.d_head = d_head

        d_embed = d_model if d_embed is None else d_embed
        self.d_embed = d_embed
        self.d_model = d_model

        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len

        self.pos_emb = RelativePositionBias(causal=True, heads=n_head)
        self.encoder = AdaptiveInput(d_model=d_model, n_classes=n_token, cutoffs=cutoffs, div_value=div_val)
        self.transformer = Feedback(
            n_layer, n_head, d_model, d_head=d_head, d_inner=d_inner, dropout=drop_out, dropatt=drop_att,
            mem_len=mem_len, seq_len=tgt_len, keep_last_hidden=False, same_length=same_length, pre_lnorm=False
        )
        self.lm_loss = AdaptiveLogSoftmax(d_model=d_model, n_classes=n_token, cutoffs=cutoffs, div_value=div_val)

        if tie_weight:
            self._share_weights()

        self._init_weights()
        self.param_dtype = next(self.parameters()).dtype

    def _batch_first(self, t):
        return torch.einsum('i...k->k...i', t)

    def _seq_first(self, t):
        return torch.einsum('ik...->ki...', t)

    def _init_weights(self):
        self.apply(weights_init)
        self.encoder.apply(weights_init)  # ensure embedding init not overridden by weight sharing

    def _share_weights(self):
        self.encoder.head.weight = self.lm_loss.head.weight
        for i in range(len(self.encoder.cutoffs) - 1):
            self.encoder.tail[i].weight = self.lm_loss.tail[i].weight

        # for i in range(len(self.encoder.cutoffs) - 1):
        #     self.encoder.tail[i][0].weight = self.lm_loss.tail[i][1].weight
        #     self.encoder.tail[i][1].weight = torch.nn.Parameter(
        #         self.lm_loss.tail[i][0].weight.transpose(0, 1)
        #     )  # sharing the projection layers

    def _reset_length(self, tgt_len, ext_len, mem_len):
        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len

    def _init_mems(self, device):
        if self.mem_len > 0:
            mems = []

            for i in range(self.transformer.n_layer + 1):
                empty = torch.empty(0, dtype=self.param_dtype, device=device)
                mems.append(empty)

            return mems
        else:
            return None

    def _update_mems(self, hids, mems, qlen, mlen):
        # does not deal with None
        if mems is None: return None

        # mems is not None
        assert len(hids) == len(mems), 'len(hids) != len(mems)'

        # There are `mlen + qlen` steps that can be cached into mems
        # For the next step, the last `ext_len` of the `qlen` tokens
        # will be used as the extended context. Hence, we only cache
        # the tokens from `mlen + qlen - self.ext_len - self.mem_len`
        # to `mlen + qlen - self.ext_len`.
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + max(0, qlen - 0 - self.ext_len)
            beg_idx = max(0, end_idx - self.mem_len)
            for i in range(len(hids)):
                cat = torch.cat([mems[i], hids[i]], dim=0)
                new_mems.append(cat[beg_idx:end_idx].detach())

        return new_mems

    def compute_loss(self, core_out, target):
        tgt_len = target.size(0)
        pred_hid = core_out[-tgt_len:]

        output = self.lm_loss(pred_hid.view(-1, pred_hid.size(-1)), target.view(-1))
        return -output.output.view(tgt_len, -1)

    def forward(self, x, y, memory=None, return_memory=False):
        b, n, device = *x.shape, x.device

        x = self.encoder(x)

        memory_keys = None
        memory_values = None

        if exists(memory) and len(memory) == 2:
            memory_keys, memory_values = memory

        outputs = []

        # calculate weighting of layers for storing to memory
        layer_weight = self.transformer.get_layer_weight()

        for x in x.split(self.mem_len, dim=1):
            dec_outp = self.transformer(
                dec_inp=x, pos_emb=self.pos_emb,
                mems=FeedbackMemory(memory_keys, memory_values),
                layer_weight=layer_weight
            )
            x = dec_outp['output']
            # agg_hiddens = dec_outp['agg_hiddens']
            memory_keys, memory_values = dec_outp['mems']

            outputs.append(x)

        x = torch.cat((outputs), dim=1)

        output = self.lm_loss(x.view(-1, x.size(-1)), y.view(-1))
        loss = -output.output.view(y.size(1), -1)

        new_memory = FeedbackMemory(memory_keys, memory_values)

        # output = {
        #     'output': self._seq_first(x),
        #     'mems': None,
        #     'loss': self._batch_first(self.compute_loss(x, target))
        # }
        #
        # if return_memory:
        #     output['mems'] = FeedbackMemory(memory_keys, memory_values)

        return loss, new_memory