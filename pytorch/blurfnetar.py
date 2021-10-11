import torch
import torch.nn as nn

from modules.adaptiveinput import AdaptiveInput
from modules.adaptivelogsoftmax import AdaptiveLogSoftmax
from modules.xlmemory import XlMemory
from fnetar import Fnetar

class BlurFnetar(nn.Module):
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
        super(BlurFnetar, self).__init__()
        self.n_token = n_token
        self.n_layer = n_layer
        self.n_head = n_head
        self.d_head = d_head

        d_embed = d_model if d_embed is None else d_embed
        self.d_embed = d_embed
        self.d_model = d_model

        self.embedder = AdaptiveInput(d_model=d_model, n_classes=n_token, cutoffs=cutoffs, div_value=div_val)
        self.transformer = Fnetar(
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

    from pytorch.utils import data_utils

    data = torch.LongTensor(data_len*B).random_(0, args.n_token).to(device)
    diter = data_utils.LMOrderedIterator(data, B, tgt_len, device=device, ext_len=ext_len)

    cutoffs = [args.n_token // 2]
    tie_projs = [False] + [True] * len(cutoffs)

    for div_val in [1, 2]:
        for d_embed in [200, 100]:
            model = Blur(args.n_token, args.n_layer, args.n_head,
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
