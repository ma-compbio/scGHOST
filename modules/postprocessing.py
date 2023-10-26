import sys
import torch
import numpy as np
import torch
# calibration
from tqdm.auto import trange

# modified function to return calibrated pairs/labels
def post_process_samples(sample_data,OEMs,nearest_neighbors,neighbor_contacts=True):
    
    all_cell_chrom_samples = sample_data

    pos_negs = [1,-1]

    all_continuous_samples = []

    num_cells = len(OEMs)
    
    for n in trange(num_cells, desc='post processing'):
        
        continuous_intra_samples = []
        all_cell_labels = all_cell_chrom_samples[n]
        
        for pn in pos_negs:

            pos_neg_intra_idx = np.where((all_cell_labels == pn))[0]
            chrm_samples = all_cell_chrom_samples[n][pos_neg_intra_idx]

            adjusted_chrm_samples = np.zeros((len(chrm_samples),3))
            chrm_samples = chrm_samples[:,:2].astype(int)#.cpu().numpy().astype(int)

            chrlen = len(OEMs[n])

            chrm_linear_samples = chrm_samples[:,0] * chrlen + chrm_samples[:,1]
            
            ms_flattened = OEMs[nearest_neighbors[n]].reshape(nearest_neighbors.shape[1],-1)
            for i in range(ms_flattened.shape[0]):
                ms_flattened[i][ms_flattened[i] > 0] = ms_flattened[i][ms_flattened[i] > 0] / np.quantile(ms_flattened[i][ms_flattened[i] > 0],0.975)

                if torch.sum(ms_flattened[i] <= 0) > 0:
                    ms_flattened[i][ms_flattened[i] <= 0] = -(ms_flattened[i][ms_flattened[i] <= 0] / np.quantile(ms_flattened[i][ms_flattened[i] <= 0],0.025))
                
                ms_flattened[i][ms_flattened[i] > 1] = 1
                ms_flattened[i][ms_flattened[i] < -1] = -1
                
            c_flattened = ms_flattened[:,chrm_linear_samples].mean(dim=0) if neighbor_contacts else ms_flattened[0,chrm_linear_samples]
            adjusted_chrm_samples[:,:2] = chrm_samples
            adjusted_chrm_samples[:,2] = c_flattened

            adjusted_del_idx = np.where(np.sign(c_flattened) != pn)[0]
            adjusted_chrm_samples = np.delete(adjusted_chrm_samples,adjusted_del_idx,axis=0)
            
            continuous_intra_samples.append(adjusted_chrm_samples)

        continuous_intra_samples = torch.tensor(np.concatenate(continuous_intra_samples)).float()
        all_continuous_samples.append(continuous_intra_samples)

    all_continuous_pairs = [all_continuous_samples[i][:,:2].long() for i in range(len(all_continuous_samples))]
    all_continuous_labels = [all_continuous_samples[i][:,2] for i in range(len(all_continuous_samples))]
    
    return all_continuous_pairs, all_continuous_labels