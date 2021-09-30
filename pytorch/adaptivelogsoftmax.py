import torch
import torch.nn.functional as F
from typing import List
from collections import namedtuple
from torch.nn import Sequential, ModuleList, Linear, Module

_ASMoutput = namedtuple('ASMoutput', ['output', 'loss'])


class AdaptiveLogSoftmax(Module):
    def __init__(
            self,
            d_model: int,
            n_classes: int,
            cutoffs: List[int],
            div_value=4.,
            head_bias=True,
            tail_drop=0.5
    ):
        super(AdaptiveLogSoftmax, self).__init__()

        cutoffs = list(cutoffs)

        if (cutoffs != sorted(cutoffs)) \
                or (min(cutoffs) <= 0) \
                or (max(cutoffs) > (n_classes - 1)) \
                or (len(set(cutoffs)) != len(cutoffs)) \
                or any([int(c) != c for c in cutoffs]):
            raise ValueError("cutoffs should be a sequence of unique, positive "
                             "integers sorted in an increasing order, where "
                             "each value is between 1 and n_classes-1")

        self.d_model = d_model
        self.n_classes = n_classes
        self.cutoffs = cutoffs + [n_classes]
        self.div_value = div_value
        self.head_bias = head_bias
        self.tail_drop = tail_drop

        self.shortlist_size = self.cutoffs[0]
        self.n_clusters = len(self.cutoffs) - 1
        self.head_size = self.shortlist_size + self.n_clusters

        self.head = Linear(self.d_model, self.shortlist_size, bias=self.head_bias)
        self.cluster = Linear(self.d_model, self.n_clusters, bias=True)
        self.tail = ModuleList()

        for i in range(self.n_clusters):

            osz = self.cutoffs[i + 1] - self.cutoffs[i]

            self.tail.append(Linear(self.d_model, osz, bias=True))

            # hsz = int(self.d_model // (self.div_value ** (i + 1)))
            # projection = Sequential(
            #     Linear(self.d_model, hsz, bias=False),
            #     Linear(hsz, osz, bias=False),
            #     nn.Dropout(self.tail_drop)
            # )
            # self.tail.append(projection)

    def reset_parameters(self):
        self.head.reset_parameters()
        for i2h, h2o in self.tail:
            i2h.reset_parameters()
            h2o.reset_parameters()

    def forward(self, input, target):
        if input.size(0) != target.size(0):
            raise RuntimeError('Input and target should have the same size '
                               'in the batch dimension.')

        used_rows = 0
        batch_size = target.size(0)

        output = input.new_zeros(batch_size)
        gather_inds = target.new_empty(batch_size)

        cutoff_values = [0] + self.cutoffs
        for i in range(len(cutoff_values) - 1):

            low_idx = cutoff_values[i]
            high_idx = cutoff_values[i + 1]

            target_mask = (target >= low_idx) & (target < high_idx)
            row_indices = target_mask.nonzero().squeeze()

            if row_indices.numel() == 0:
                continue

            if i == 0:
                gather_inds.index_copy_(0, row_indices, target[target_mask])

            else:
                relative_target = target[target_mask] - low_idx
                input_subset = input.index_select(0, row_indices)

                cluster_output = self.tail[i - 1](input_subset)
                cluster_index = self.shortlist_size + i - 1

                gather_inds.index_fill_(0, row_indices, cluster_index)

                cluster_logprob = F.log_softmax(cluster_output, dim=1)
                local_logprob = cluster_logprob.gather(1, relative_target.unsqueeze(1))
                output.index_copy_(0, row_indices, local_logprob.squeeze(1))

            used_rows += row_indices.numel()

        if used_rows != batch_size:
            raise RuntimeError("Target values should be in [0, {}], "
                               "but values in range [{}, {}] "
                               "were found. ".format(self.n_classes - 1,
                                                     target.min().item(),
                                                     target.max().item()))

        head_output = torch.cat([self.head(input), self.cluster(input)], dim=-1)
        head_logprob = F.log_softmax(head_output, dim=1)
        output += head_logprob.gather(1, gather_inds.unsqueeze(1)).squeeze()
        loss = (-output).mean()

        return _ASMoutput(output, loss)

    def _get_full_log_prob(self, input, head_output):
        """ Given input tensor, and output of `self.head`,
        compute the log of the full distribution """

        out = input.new_empty((head_output.size(0), self.n_classes))
        head_logprob = F.log_softmax(head_output, dim=1)

        out[:, :self.shortlist_size] = head_logprob[:, :self.shortlist_size]

        for i, (start_idx, stop_idx) in enumerate(zip(self.cutoffs, self.cutoffs[1:])):
            cluster_output = self.tail[i](input)
            cluster_logprob = F.log_softmax(cluster_output, dim=1)
            output_logprob = cluster_logprob + head_logprob[:, self.shortlist_size + i].unsqueeze(1)

            out[:, start_idx:stop_idx] = output_logprob

        return out

    def log_prob(self, input):
        r""" Computes log probabilities for all :math:`\texttt{n\_classes}`
        Args:
            input (Tensor): a minibatch of examples
        Returns:
            log-probabilities of for each class :math:`c`
            in range :math:`0 <= c <= \texttt{n\_classes}`, where :math:`\texttt{n\_classes}` is a
            parameter passed to ``AdaptiveLogSoftmaxWithLoss`` constructor.
        Shape:
            - Input: :math:`(N, \texttt{in\_features})`
            - Output: :math:`(N, \texttt{n\_classes})`
        """

        head_output = torch.cat([self.head(input), self.cluster(input)], dim=-1)
        return self._get_full_log_prob(input, head_output)

    def predict(self, input):
        r""" This is equivalent to `self.log_pob(input).argmax(dim=1)`,
        but is more efficient in some cases.
        Args:
            input (Tensor): a minibatch of examples
        Returns:
            output (Tensor): a class with the highest probability for each example
        Shape:
            - Input: :math:`(N, \texttt{in\_features})`
            - Output: :math:`(N)`
        """

        head_output = torch.cat([self.head(input), self.cluster(input)], dim=-1)
        output = torch.argmax(head_output, dim=1)
        not_in_shortlist = (output >= self.shortlist_size)
        all_in_shortlist = not (not_in_shortlist.any())

        if all_in_shortlist:
            return output

        elif not_in_shortlist.all():
            log_prob = self._get_full_log_prob(input, head_output)
            return torch.argmax(log_prob, dim=1)

        else:
            log_prob = self._get_full_log_prob(input[not_in_shortlist],
                                               head_output[not_in_shortlist])
            output[not_in_shortlist] = torch.argmax(log_prob, dim=1)
            return output