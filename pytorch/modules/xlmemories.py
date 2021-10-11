import torch
import torch.nn as nn

from modules.xlmemory import XlMemory

class XlMemories(nn.Module):
    def __init__(
            self, n_stream: int,
            n_layer: int,
            tgt_len: int,
            mem_len: int,
            ext_len: int,
            dtype: torch.dtype
    ):
        assert mem_len >= tgt_len, 'l_k !>= l_q'
        super(XlMemories, self).__init__()

        self.memories = [
            XlMemory(n_layer=n_layer, tgt_len=tgt_len, mem_len=mem_len, ext_len=ext_len, dtype=dtype)
            for _ in range(n_stream)
        ]

    def __len__(self):
        return len(self.memories)

    def __getitem__(self, item: int) -> torch.Tensor:
        return self.memories[item]

    def __setitem__(self, key: int, value: XlMemory):
        self.memories[key] = value

    def __iter__(self):
        return iter(self.memories)

    def update_memory_stream(self, stream_index: int, memory: XlMemory):
        self.memories[stream_index] = memory