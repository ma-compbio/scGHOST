# parallelized constrained random walks on the GPU with random walks to inter corr matrix
import os
import torch
import pickle
import gc
import numpy as np

from utilities.gpu import to_cuda
from modules.preprocessing import parse_chromosomes, parse_cell_types
from torch.nn import functional as F
from tqdm import trange

def sample_chrom(chrom_num,OEMs,runtime_args,use_breakpoint=False):
    
    all_cell_chrom_samples = []
    all_cell_pairs = []
    all_cell_labels = []
    all_cell_intra_inter = []
    all_cell_chrms = []
    all_cell_chrm_offsets = []
    
    num_cells = len(OEMs)

    num_walks = runtime_args['random_walk']['num_walks']
    ignore_top = runtime_args['random_walk']['ignore_top']
    top_percentile = runtime_args['random_walk']['top_percentile']

    for cnum in trange(num_cells):
        
        chrm_offset = 0

        m = to_cuda(torch.tensor(np.nan_to_num(np.corrcoef(OEMs[cnum]))).float())

        bpt = 0
        
        if use_breakpoint:
            bpt = len(m) // 2
            m = m[:bpt,bpt:]
        
        all_samples = []

        num_ignore = np.max([int(m.shape[0] * ignore_top),1])
        
        num_top = int(m.shape[0] * top_percentile)

        sorted_slc_w,sorted_slc_i = m.sort(dim=1)
        sorted_slc_w_T,sorted_slc_i_T = m.T.sort(dim=1)

        for n in range(num_walks):

            test_samples = to_cuda(torch.arange(m.shape[0]))

            w1,i1 = sorted_slc_w[test_samples],sorted_slc_i[test_samples]

            pw1 = torch.exp(w1[:,-num_top-num_ignore:-num_ignore])
            nw1 = torch.exp(w1[:,num_ignore:num_top+num_ignore])

            pi1 = i1[:,-num_top-num_ignore:-num_ignore]
            ni1 = i1[:,num_ignore:num_top+num_ignore]
            
            p_mask = F.one_hot(torch.multinomial(pw1,1).flatten(),num_classes=pi1.shape[1])
            n_mask = F.one_hot(torch.multinomial(nw1,1).flatten(),num_classes=ni1.shape[1])

            pos_selection1 = (pi1 * p_mask).sum(dim=1)
            neg_selection1 = (ni1 * n_mask).sum(dim=1)

            pw2,pi2 = sorted_slc_w_T[pos_selection1],sorted_slc_i_T[pos_selection1]
            
            pw2 = torch.exp(pw2[:,-num_top-num_ignore:-num_ignore])
            pi2 = pi2[:,-num_top-num_ignore:-num_ignore]

            nw2,ni2 = sorted_slc_w_T[neg_selection1],sorted_slc_i_T[neg_selection1]
            
            nw2 = torch.exp(nw2[:,-num_top-num_ignore:-num_ignore])
            ni2 = ni2[:,-num_top-num_ignore:-num_ignore]

            p_mask = F.one_hot(torch.multinomial(pw2,1).flatten(),num_classes=pi2.shape[1])
            n_mask = F.one_hot(torch.multinomial(nw2,1).flatten(),num_classes=ni2.shape[1])

            pos_selection2 = (pi2 * p_mask).sum(dim=1)
            neg_selection2 = (ni2 * n_mask).sum(dim=1)
            
            selections = torch.stack((
                pos_selection1 + bpt,
                pos_selection2,
                neg_selection1 + bpt,
                neg_selection2
            )).T.flatten()
            
            labels = to_cuda(torch.tensor([1,1,-1,-1]).repeat(len(test_samples)))

            interactions = torch.stack((
                test_samples.repeat_interleave(4) + chrm_offset,
                selections,
                labels,
                to_cuda(torch.ones(len(selections))),
                to_cuda(torch.ones(len(selections))) * chrom_num,
                to_cuda(torch.ones(len(selections)))
            )).T

            interactions = torch.cat((
                interactions,
                torch.stack((
                    torch.stack((pos_selection1 + bpt,neg_selection1 + bpt)).T.flatten(),
                    torch.stack((pos_selection2,neg_selection2)).T.flatten(),
                    to_cuda(torch.ones(len(pos_selection1)+len(neg_selection1))),
                    to_cuda(torch.ones(len(pos_selection1) + len(neg_selection1))),
                    to_cuda(torch.ones(len(pos_selection1) + len(neg_selection1))) * chrom_num,
                    to_cuda(torch.ones(len(pos_selection1) + len(neg_selection1)))
                )).T
            ))

            all_samples.append(interactions)

        all_samples = torch.unique(torch.cat(all_samples),dim=0)
        all_pairs = all_samples[:,:2].long()
        all_labels = all_samples[:,2].float()
        all_intra_inter = all_samples[:,3].long()
        all_chrms = all_samples[:,4].long()
        
        all_cell_chrom_samples.append(all_samples)
        all_cell_pairs.append(all_pairs)
        all_cell_labels.append(all_labels)
        all_cell_intra_inter.append(all_intra_inter)
        all_cell_chrms.append(all_chrms)
        gc.collect()

    return all_cell_chrom_samples,all_cell_labels,all_cell_pairs,all_cell_intra_inter

def random_walk(runtime_args):

    chromosomes = parse_chromosomes(runtime_args)
    
    for chrom_num in chromosomes:

        print('Processing random walks in chromosome %d' % chrom_num)
        
        oe_path = os.path.join(runtime_args['data_directory'],'chr%d_oe.npy' % chrom_num)
        out_path = os.path.join(runtime_args['data_directory'],'chr%d_samples.npy' % chrom_num)

        OEMs = np.load(oe_path)

        all_cell_chrom_samples,all_cell_labels,all_cell_pairs,all_cell_intra_inter = sample_chrom(chrom_num,OEMs,runtime_args)

        pickle.dump({
            'chrom_samples' : all_cell_chrom_samples,
            'chrom_pairs' : all_cell_pairs,
            'chrom_labels' :  all_cell_labels,
            'chrom_intra_inter' : [np.ones(len(all_cell_chrom_samples[i])) for i in range(len(all_cell_intra_inter))],
            'chrom_chrms' : [chrom_num * np.ones(len(all_cell_chrom_samples[i])) for i in range(len(all_cell_chrom_samples))],
        }, open(out_path,'wb'))
        
        del all_cell_chrom_samples,all_cell_labels,all_cell_pairs,all_cell_intra_inter
        torch.cuda.empty_cache()