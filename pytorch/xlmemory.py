import torch

class XlMemory:
    def __init__(self, n_layer: int, tgt_len: int, mem_len: int, ext_len: int, dtype: torch.dtype):
        assert mem_len >= tgt_len, 'l_k !>= l_q'
        self.n_layer = n_layer

        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len

        self.dtype = dtype

        self.klen = tgt_len + mem_len
        self.memory = [torch.empty(0, dtype=self.dtype) for _ in range(self.n_layer + 1)]

    def __len__(self):
        return len(self.memory)

    def __getitem__(self, item: int) -> torch.Tensor:
        return self.memory[item]

    def __setitem__(self, key: int, value: torch.Tensor):
        self.memory[key] = value

    def __iter__(self):
        return iter(self.memory)

    def size(self, i: int):
        return self.memory[i].size(1) if len(self.memory[i].size()) > 1 else 0

    def update_memory(self, hids, mems, qlen, mlen):
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
                cat = torch.cat([mems[i].to(hids[i].device), hids[i]], dim=1)
                new_mems.append(cat[:, beg_idx:end_idx].detach())

        self.memory = new_mems