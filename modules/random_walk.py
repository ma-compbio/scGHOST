# parallelized constrained random walks on the GPU with random walks to inter corr matrix
import os
import torch
import pickle
import gc
import numpy as np

from utilities.gpu import to_cuda
from utilities.parsers import parse_nearest_neighbors
from utilities.helper import random_sample
from modules.preprocessing import parse_chromosomes
from torch.nn import functional as F
from tqdm import trange

def sample_chrom(chrom_num,OEMs,cell_range,nearest_neighbors,num_walks=50,use_breakpoint=False):
    
    all_cell_chrom_samples = []
    layered_maps = OEMs[nearest_neighbors[cell_range]]

    for cnum in trange(len(cell_range)):
        chrm_offset = 0

        # m = to_cuda(torch.tensor(torch.nan_to_num(layered_maps[cnum])).float())
        m = to_cuda(layered_maps[cnum].float()) # cast from bfloat16 to float32 for precision with sorting

        bpt = 0
        
        if use_breakpoint:
            bpt = len(m) // 2
            m = m[:bpt,bpt:]
        
        all_samples = []

        num_top = int(m.shape[1] * 0.25)

        sorted_slc_w = torch.zeros_like(m)
        sorted_slc_i = torch.zeros_like(m)
        sorted_slc_w_T = torch.zeros_like(m)
        sorted_slc_i_T = torch.zeros_like(m)
        
        for i in range(m.shape[0]):
            sorted_slc_w[i],sorted_slc_i[i] = m[i].sort(dim=1)
            sorted_slc_w_T[i],sorted_slc_i_T[i] = m[i].T.sort(dim=1)

        sorted_slc_w = sorted_slc_w.repeat(num_walks,1,1,1)
        sorted_slc_w_T = sorted_slc_w_T.repeat(num_walks,1,1,1)
        sorted_slc_i = sorted_slc_i.repeat(num_walks,1,1,1)
        sorted_slc_i_T = sorted_slc_i_T.repeat(num_walks,1,1,1)

        test_samples = to_cuda(torch.arange(m.shape[1]))

        w1,i1 = sorted_slc_w[:,:,test_samples],sorted_slc_i[:,:,test_samples]

        pw1 = torch.exp(w1[...,-num_top:])
        nw1 = 1/torch.exp(w1[...,:num_top]) # inverse to select for lower contact frequencies

        pi1 = i1[...,-num_top:]
        ni1 = i1[...,:num_top]

        p_mask = F.one_hot(torch.squeeze(random_sample(pw1,1)[1]),num_classes=pi1.shape[-1])
        n_mask = F.one_hot(torch.squeeze(random_sample(nw1,1)[1]),num_classes=ni1.shape[-1])

        pos_selection1 = ((pi1 * p_mask).sum(dim=-1)).long()
        neg_selection1 = ((ni1 * n_mask).sum(dim=-1)).long()

        pw2 = torch.gather(sorted_slc_w_T,-2,pos_selection1[...,None].tile(1,1,1,pos_selection1.shape[-1]))
        pi2 = torch.gather(sorted_slc_i_T,-2,pos_selection1[...,None].tile(1,1,1,pos_selection1.shape[-1]))

        pw2 = torch.exp(pw2[...,-num_top:])
        pi2 = pi2[...,-num_top:]

        nw2 = torch.gather(sorted_slc_w_T,-2,neg_selection1[...,None].tile(1,1,1,pos_selection1.shape[-1]))
        ni2 = torch.gather(sorted_slc_i_T,-2,neg_selection1[...,None].tile(1,1,1,pos_selection1.shape[-1]))

        nw2 = torch.exp(nw2[...,-num_top:])
        ni2 = ni2[...,-num_top:]

        p_mask = F.one_hot(torch.squeeze(random_sample(pw2,1)[1]),num_classes=pi2.shape[-1])
        n_mask = F.one_hot(torch.squeeze(random_sample(nw2,1)[1]),num_classes=ni2.shape[-1])

        pos_selection2 = ((pi2 * p_mask).sum(dim=-1)).long()
        neg_selection2 = ((ni2 * n_mask).sum(dim=-1)).long()

        for i in range(num_walks):
            selections = to_cuda(torch.stack((
                pos_selection1[i].flatten() + bpt,
                pos_selection2[i].flatten(),
                neg_selection1[i].flatten() + bpt,
                neg_selection2[i].flatten()
            )).T.flatten())
            
            labels = to_cuda(torch.tensor([1,1,-1,-1]).repeat(len(m) * len(test_samples)))

            interactions = torch.stack((
                test_samples.repeat_interleave(4 * len(m)) + chrm_offset,
                selections,
                labels,
            )).T

            all_samples.append(interactions)

        all_samples = torch.unique(torch.cat(all_samples),dim=0).cpu().numpy().astype(np.int16)
        all_cell_chrom_samples.append(all_samples)

    return all_cell_chrom_samples