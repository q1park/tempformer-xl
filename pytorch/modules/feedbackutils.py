import torch

def exists(val):
    return val is not None


def safe_cat(arr, el, dim=1):
    if not exists(arr):
        return el
    return torch.cat((arr.detach(), el), dim=dim)