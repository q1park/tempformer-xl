import torch

def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def safe_cat(arr, el, dim=1):
    if not exists(arr):
        return el
    return torch.cat((arr, el), dim=dim)