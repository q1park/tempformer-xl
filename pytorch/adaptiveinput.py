import torch.nn as nn


class AdaptiveInput(nn.Module):
    def __init__(self, d_model, n_classes, cutoffs=None, div_value=2.0, head_bias=False, tail_drop=0.5):
        super(AdaptiveInput, self).__init__()
        if not cutoffs:
            cutoffs = [5000, 10000]
        cutoffs = list(cutoffs)

        if (cutoffs != sorted(cutoffs)) \
                or (min(cutoffs) <= 0) \
                or (max(cutoffs) >= (n_classes - 1)) \
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

        self.n_clusters = len(self.cutoffs) - 1
        self.head_size = self.cutoffs[0]

        # self.head = nn.Sequential(nn.Embedding(self.head_size, self.in_features),
        #                           nn.Linear(self.in_features, self.in_features, bias=self.head_bias))
        self.head = nn.Embedding(self.head_size, self.d_model)

        self.tail = nn.ModuleList()

        for i in range(self.n_clusters):
            osz = self.cutoffs[i + 1] - self.cutoffs[i]
            self.tail.append(nn.Embedding(osz, self.d_model))

            # hsz = int(self.d_model // (self.div_value ** (i + 1)))
            # projection = nn.Sequential(
            #     nn.Embedding(osz, hsz),
            #     nn.Linear(hsz, self.d_model, bias=False),
            #     nn.Dropout(self.tail_drop)
            # )
            # self.tail.append(projection)

    def forward(self, input):
        used_rows = 0
        input_size = list(input.size())

        output = input.new_zeros([input.size(0) * input.size(1)] + [self.d_model]).float()
        input = input.view(-1)

        cutoff_values = [0] + self.cutoffs
        for i in range(len(cutoff_values) - 1):

            low_idx = cutoff_values[i]
            high_idx = cutoff_values[i + 1]

            input_mask = (input >= low_idx) & (input < high_idx)
            row_indices = input_mask.nonzero().squeeze()

            if row_indices.numel() == 0:
                continue
            out = self.head(input[input_mask] - low_idx) if i == 0 else self.tail[i - 1](input[input_mask] - low_idx)
            output.index_copy_(0, row_indices, out)
            used_rows += row_indices.numel()

        return output.view(input_size[0], input_size[1], -1)

