import torch
import torch.nn as nn
from collections import namedtuple

FeedbackMemory = namedtuple('Memory', ['keys', 'values'])

class FeedbackMemories(nn.Module):
    def __init__(
            self, n_stream: int
    ):
        super(FeedbackMemories, self).__init__()

        self.memories = [
            FeedbackMemory(None, None)
            for _ in range(n_stream)
        ]

    def __len__(self):
        return len(self.memories)

    def __getitem__(self, item: int) -> torch.Tensor:
        return self.memories[item]

    def __setitem__(self, key: int, value: FeedbackMemory):
        self.memories[key] = value

    def __iter__(self):
        return iter(self.memories)

    def update_memory_stream(self, stream_index: int, memory: FeedbackMemory):
        self.memories[stream_index] = memory