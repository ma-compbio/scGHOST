# parallelized constrained random walks on the GPU with random walks to inter corr matrix
import os
import torch
import pickle
import gc
import numpy as np

from utilities.gpu import to_cuda
from utilities.parsers import parse_chromosomes, parse_nearest_neighbors
from torch.nn import functional as F
from tqdm import trange

def sample_chrom(chrom_num,OEMs,cell_range,nearest_neighbors,use_breakpoint=False):
    
    all_cell_chrom_samples = []
    layered_maps = OEMs[nearest_neighbors[cell_range]]

    for cnum in trange(len(cell_range)):
        chrm_offset = 0

        m = to_cuda(torch.tensor(np.nan_to_num(layered_maps[cnum])).float())

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
        num_walks = 50 #hard-coding from the config file
        for n in range(num_walks):

            test_samples = to_cuda(torch.arange(m.shape[1]))

            w1,i1 = sorted_slc_w[:,test_samples],sorted_slc_i[:,test_samples]

            pw1 = torch.exp(w1[:,:,-num_top:])
            nw1 = 1/torch.exp(w1[:,:,:num_top]) # inverse to select for lower contact frequencies

            pi1 = i1[:,:,-num_top:]
            ni1 = i1[:,:,:num_top]

            p_mask = torch.stack(
                [F.one_hot(torch.multinomial(pw1[i],1).flatten(),num_classes=pi1.shape[-1]) for i in range(len(pw1))]
            )
            n_mask = torch.stack(
                [F.one_hot(torch.multinomial(nw1[i],1).flatten(),num_classes=ni1.shape[-1]) for i in range(len(nw1))]
            )
            
            pos_selection1 = ((pi1 * p_mask).sum(dim=-1)).type(torch.LongTensor)
            neg_selection1 = ((ni1 * n_mask).sum(dim=-1)).type(torch.LongTensor)

            pw2 = torch.stack(
                [sorted_slc_w_T[i,pos_selection1[i]] for i in range(len(pos_selection1))]
            )
            pi2 = torch.stack(
                [sorted_slc_i_T[i,pos_selection1[i]] for i in range(len(pos_selection1))]
            )
            
            pw2 = torch.exp(pw2[:,:,-num_top:])
            pi2 = pi2[:,:,-num_top:]

            
            nw2 = torch.stack([sorted_slc_w_T[i,neg_selection1[i]] for i in range(len(neg_selection1))])
            ni2 = torch.stack([sorted_slc_i_T[i,neg_selection1[i]] for i in range(len(neg_selection1))])
            
            nw2 = torch.exp(nw2[:,:,-num_top:])
            ni2 = ni2[:,:,-num_top:]

            p_mask = torch.stack(
                [F.one_hot(torch.multinomial(pw2[i],1).flatten(),num_classes=pi2.shape[-1]) for i in range(len(pw2))]
            )
            n_mask = torch.stack(
                [F.one_hot(torch.multinomial(nw2[i],1).flatten(),num_classes=ni2.shape[-1]) for i in range(len(nw2))]
            )

            pos_selection2 = ((pi2 * p_mask).sum(dim=-1)).type(torch.LongTensor)
            neg_selection2 = ((ni2 * n_mask).sum(dim=-1)).type(torch.LongTensor)
            
            selections = to_cuda(torch.stack((
                pos_selection1.flatten() + bpt,
                pos_selection2.flatten(),
                neg_selection1.flatten() + bpt,
                neg_selection2.flatten()
            )).T.flatten())
            
            labels = to_cuda(torch.tensor([1,1,-1,-1]).repeat(len(m) * len(test_samples)))

            interactions = torch.stack((
                test_samples.repeat_interleave(4 * len(m)) + chrm_offset,
                selections,
                labels,
            )).T

            all_samples.append(interactions)

        all_samples = torch.unique(torch.cat(all_samples),dim=0)
        all_cell_chrom_samples.append(all_samples.cpu())
        
        gc.collect()
        torch.cuda.empty_cache()

    return all_cell_chrom_samples

def random_walk(runtime_args):

    chromosomes = parse_chromosomes(runtime_args)
    nearest_neighbors = parse_nearest_neighbors(runtime_args)

    for chrom_num in chromosomes:

        print('Processing random walks in chromosome %d' % chrom_num)
        
        oe_path = os.path.join(runtime_args['data_directory'],'chr%d_oe.npy' % chrom_num)
        out_path = os.path.join(runtime_args['data_directory'],'chr%d_samples.npy' % chrom_num)

        OEMs = np.load(oe_path)

        corr_OEMs = np.zeros_like(OEMs)

        for i in trange(len(OEMs)):
            corr_OEMs[i] = np.corrcoef(OEMs[i])
            np.fill_diagonal(corr_OEMs[i],0)
        
        all_cell_chrom_samples = sample_chrom(chrom_num,corr_OEMs,np.arange(len(corr_OEMs)),nearest_neighbors)
        
        pickle.dump({
            'chrom_samples' : all_cell_chrom_samples,
        }, open(out_path,'wb'))

        del all_cell_chrom_samples
        torch.cuda.empty_cache()
