import os, sys
import glob

from collections import Counter, OrderedDict
import numpy as np
import torch

from utils.vocabulary import Vocab

class LMOrderedIterator(object):
    def __init__(self, data, bsz, bptt, device='cpu', ext_len=None):
        """
            data -- LongTensor -- the LongTensor is strictly ordered
        """
        self.bsz = bsz
        self.bptt = bptt
        self.ext_len = ext_len if ext_len is not None else 0

        self.device = device

        # Work out how cleanly we can divide the dataset into bsz parts.
        self.n_step = data.size(0) // bsz

        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, self.n_step * bsz)

        # Evenly divide the data across the bsz batches.
        self.data = data.view(bsz, -1).contiguous().to(device)

        # Number of mini-batches
        self.n_batch = (self.n_step + self.bptt - 1) // self.bptt

    def get_batch(self, i, bptt=None):
        if bptt is None:
            bptt = self.bptt

        seq_len = min(bptt, self.data.size(1) - 1 - i)

        end_idx = i + seq_len
        beg_idx = max(0, i - self.ext_len)

        data = self.data[:, beg_idx:end_idx].contiguous()
        target = self.data[:, i+1:i+1+seq_len].contiguous()

        return data, target, seq_len

    def get_fixlen_iter(self, start=0):
        for i in range(start, self.data.size(1) - 1, self.bptt):
            yield self.get_batch(i)

    def __iter__(self):
        return self.get_fixlen_iter()

class Corpus(object):
    def __init__(self, path, dataset, *args, **kwargs):
        self.dataset = dataset
        self.vocab = Vocab(*args, **kwargs)

        self.vocab.count_file(os.path.join(path, 'train.txt'))
        self.vocab.build_vocab()

        self.train = self.vocab.encode_file(
            os.path.join(path, 'train.txt'), ordered=True)
        self.valid = self.vocab.encode_file(
            os.path.join(path, 'valid.txt'), ordered=True)
        self.test = self.vocab.encode_file(
            os.path.join(path, 'test.txt'), ordered=True)

    def get_iterator(self, split, *args, **kwargs):
        if split == 'train':
            data_iter = LMOrderedIterator(self.train, *args, **kwargs)

        elif split in ['valid', 'test']:
            data = self.valid if split == 'valid' else self.test
            data_iter = LMOrderedIterator(data, *args, **kwargs)

        return data_iter


def get_lm_corpus(datadir, dataset):
    fn = os.path.join(datadir, 'cache.pt')
    if os.path.exists(fn):
        print('Loading cached dataset...')
        corpus = torch.load(fn)
    else:
        print('Producing dataset {}...'.format(dataset))
        kwargs = {}
        if dataset in ['wt103', 'wt2']:
            kwargs['special'] = ['<eos>']
            kwargs['lower_case'] = False

        corpus = Corpus(datadir, dataset, **kwargs)
        torch.save(corpus, fn)

    return corpus

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='unit test')
    parser.add_argument('--datadir', type=str, default='../data/text8',
                        help='location of the data corpus')
    parser.add_argument('--dataset', type=str, default='text8',
                        choices=['ptb', 'wt2', 'wt103', 'lm1b', 'enwik8', 'text8'],
                        help='dataset name')
    args = parser.parse_args()

    corpus = get_lm_corpus(args.datadir, args.dataset)
    print('Vocab size : {}'.format(len(corpus.vocab.idx2sym)))
