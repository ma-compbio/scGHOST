import os
import torch
import numpy as np
import pickle
import gc

from tqdm.auto import trange
from modules.preprocessing import parse_chromosomes
from torch import nn
from tqdm import trange
from torch.nn import functional as F

num_cells = 500

def to_cuda(x):
    if torch.cuda.is_available():
        return x.cuda()

    return x


class hubs(nn.Module):
    def __init__(self, N, num_cells, hidden_dim=128):
        super(hubs, self).__init__()
        self.N = N
        self.num_cells = num_cells
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(self.N * self.num_cells,
                                      self.hidden_dim, sparse=True, max_norm=1)

        to_cuda(self)

    def to_one_hot(self, x):
        return F.one_hot(x, num_classes=self.N * self.num_cells).float()

    def embed(self, x):
        return self.embedding(x)


def prep_pairs_labels(all_pairs, all_labels, gap, indices=None, thresh=None):
    concatenated_pairs = []
    concatenated_labels = []

    lengths = [len(xx) for xx in all_pairs]
    clip_len = np.min(lengths) if thresh is None else thresh
    iterable = range(len(all_pairs)) if indices is None else indices

    kept_cells = []
    n = 0

    # Random permutation within each cell
    for i in iterable:
        cell_pairs = all_pairs[i]
        
        if thresh is not None and len(cell_pairs) < thresh:
            continue

        id_ = torch.randperm(len(cell_pairs))[:clip_len]
        concatenated_pairs.append(cell_pairs[id_] + n * gap)
        concatenated_labels.append(all_labels[i][id_])

        n += 1
        kept_cells.append(i)
        
    # Stack instead of concat, shape of (#cell, #pairs, 2)
    concatenated_pairs = torch.stack(concatenated_pairs, dim=0)
    concatenated_labels = torch.stack(concatenated_labels, dim=0)

    return (concatenated_pairs, concatenated_labels) if thresh is None else (concatenated_pairs,concatenated_labels,np.array(kept_cells))


def embed_single_cells_unified(all_continuous_pairs, all_continuous_labels, OEMs, embedding_file, epochs=1,
                               cell_nums=None, batch_size=64, verbose=False, prepped=False):
    cell_nums = np.arange(len(all_continuous_pairs)) if cell_nums is None else cell_nums

    model = hubs(len(OEMs[0]), len(cell_nums), hidden_dim=128)
    bs = batch_size

    all_Es = []
    optimizer = torch.optim.SparseAdam(model.parameters())

    if not prepped:
        all_continuous_pairs, all_continuous_labels = prep_pairs_labels(all_continuous_pairs,
                                                                      all_continuous_labels,
                                                                      OEMs[0].shape[0],
                                                                      indices=cell_nums)
    for epoch in range(epochs):

        shuffle_id = torch.randperm(all_continuous_pairs.shape[1])

        N_pairs = all_continuous_pairs.shape[-2]
        rloss = 0
        rsamples = 0
        bar = trange(0, N_pairs, bs) if verbose else range(0, N_pairs, bs)

        for i in bar:
            # During training, sample a batch of pairs from each cell (can be small 16 yields good results)
            # You can also sample a batch of cells as well, but that needs to be sth large, like 2k cells etc.
            x = model.embed(to_cuda(all_continuous_pairs[:, shuffle_id[i:i + bs], :]))
            x1 = x[:, :, 0]
            x2 = x[:, :, 1]

            y = to_cuda(all_continuous_labels[:, shuffle_id[i:i + bs]])
            sim = F.cosine_similarity(x1, x2, dim=-1)
            optimizer.zero_grad()
            loss = F.mse_loss(sim, y)
            loss.backward()
            optimizer.step()

            blen = x.shape[1]
            rloss += float(loss) * blen
            rsamples += blen

        print('Epoch %d: %d/%d -- %.6f loss' % (epoch, rsamples, N_pairs, rloss / rsamples), end='\r')

        print()

    num_loci = model.N * model.num_cells
    bar = trange(0, num_loci, bs) if verbose else range(0, num_loci, bs)

    for i in bar:
        end = np.min([i + bs, num_loci])

        x = model.embed(to_cuda(torch.arange(i, end))).to_dense().detach().cpu().numpy()
        all_Es.append(x)

    all_Es = np.vstack(all_Es)  # final shape - (num_cells * num_chrom_loci, hidden_dim)

    all_Es = all_Es.reshape((model.num_cells, model.N, model.hidden_dim))

    np.save(embedding_file, all_Es)