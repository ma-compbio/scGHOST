import os
import torch
import gc
import numpy as np
import pickle

from utilities.parsers import parse_chromosomes, parse_nearest_neighbors
from tqdm import trange
from torch.nn import functional as F

def to_cuda(x):
    
    if torch.cuda.is_available():
        return x.cuda()
    
    return x

def post_process_samples(sample_file,OEM_file,save_path,nearest_neighbors):
    sample_data = pickle.load(open(sample_file,'rb'))
    OEMs = np.load(OEM_file)
    
    all_cell_chrom_samples = sample_data['chrom_samples']

    pos_negs = [1,-1]

    all_continuous_samples = []

    num_cells = len(OEMs)
    
    for n in trange(num_cells):
        
        continuous_intra_samples = []
        all_cell_labels = all_cell_chrom_samples[n]
        
        for pn in pos_negs:

            pos_neg_intra_idx = torch.where((all_cell_labels == pn))[0]
            chrm_samples = all_cell_chrom_samples[n][pos_neg_intra_idx]

            adjusted_chrm_samples = np.zeros((len(chrm_samples),3))
            chrm_samples = chrm_samples[:,:2].cpu().numpy().astype(int)

            chrlen = len(OEMs[n])

            chrm_linear_samples = chrm_samples[:,0] * chrlen + chrm_samples[:,1]
            
            ms_flattened = OEMs[nearest_neighbors[n]].reshape(nearest_neighbors.shape[1],-1)
            for i in range(ms_flattened.shape[0]):
                ms_flattened[i][ms_flattened[i] > 0] = ms_flattened[i][ms_flattened[i] > 0] / np.quantile(ms_flattened[i][ms_flattened[i] > 0],0.975)
            
                if np.sum(ms_flattened[i] <= 0) > 0:
                    ms_flattened[i][ms_flattened[i] <= 0] = -(ms_flattened[i][ms_flattened[i] <= 0] / np.quantile(ms_flattened[i][ms_flattened[i] <= 0],0.025))
                
                ms_flattened[i][ms_flattened[i] > 1] = 1
                ms_flattened[i][ms_flattened[i] < -1] = -1
                
            c_flattened = ms_flattened[0,chrm_linear_samples]
            
            adjusted_chrm_samples[:,:2] = chrm_samples
            adjusted_chrm_samples[:,2] = c_flattened

            adjusted_del_idx = np.where(np.sign(c_flattened) != pn)[0]
            adjusted_chrm_samples = np.delete(adjusted_chrm_samples,adjusted_del_idx,axis=0)
            
            continuous_intra_samples.append(adjusted_chrm_samples)

        continuous_intra_samples = torch.tensor(np.concatenate(continuous_intra_samples)).float()
        all_continuous_samples.append(continuous_intra_samples)

    all_continuous_pairs = [all_continuous_samples[i][:,:2].long() for i in range(len(all_continuous_samples))]
    all_continuous_labels = [all_continuous_samples[i][:,2] for i in range(len(all_continuous_samples))]
    
    pickle.dump({
        'pairs' : all_continuous_pairs,
        'labels' : all_continuous_labels
    },open(save_path,'wb'))
    
    gc.collect()

def label_calibration(runtime_args):

    chromosomes = parse_chromosomes(runtime_args)
    nearest_neighbors = parse_nearest_neighbors(runtime_args)
    
    for chrn in range(chromosomes):
        
        sample_path = os.path.join(runtime_args['data_directory'],'chr%d_samples.pkl' % chrn)
        oe_path = os.path.join(runtime_args['data_directory'],'chr%d_oe.npy' % chrn)
        out_path = os.path.join(runtime_args['data_directory'],'chr%d_calibrated_samples.pkl' % chrn)

        print('Processing chromosome %d' % chrn)
        post_process_samples(
            sample_path,
            oe_path,
            out_path,
            nearest_neighbors
        )
        
        torch.cuda.empty_cache()
