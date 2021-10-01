import torch
import torch.nn as nn

class XlPosition(nn.Module):
    def __init__(self, d_model: int, n_head: int, d_head: int, clamp_len: int):
        super(XlPosition, self).__init__()
        self.d_model = d_model
        self.clamp_len = clamp_len
        self.bias = nn.ParameterDict({
            'k': nn.Parameter(torch.Tensor(n_head, d_head)),
            'r': nn.Parameter(torch.Tensor(n_head, d_head))
        })

        inv_freq = 1 / (10000 ** (torch.arange(0.0, d_model, 2.0) / d_model))
        self.register_buffer('inv_freq', inv_freq)

    def wave_grid(self, klen: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        pos_sequence = torch.arange(klen - 1, -1, -1.0, device=device, dtype=dtype)

        if self.clamp_len > 0:
            pos_sequence.clamp_(max=self.clamp_len)

        sinusoidal = torch.outer(pos_sequence, self.inv_freq)
        pos_grid = torch.cat([sinusoidal.sin(), sinusoidal.cos()], dim=-1)

        return pos_grid[:, :]