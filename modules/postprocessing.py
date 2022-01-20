import os
import torch
import gc
import numpy as np
import pickle

from modules.preprocessing import parse_chromosomes
from tqdm import trange
from torch.nn import functional as F

def to_cuda(x):
    
    if torch.cuda.is_available():
        return x.cuda()
    
    return x

def post_process_samples(sample_file,OEM_file,save_path):
    sample_data = pickle.load(open(sample_file,'rb'))
    OEMs = np.load(OEM_file)
    
    all_cell_chrom_samples = sample_data['chrom_samples']
    all_cell_pairs = sample_data['chrom_pairs']
    all_cell_labels = sample_data['chrom_labels']
    all_cell_intra_inter = sample_data['chrom_intra_inter']
    all_cell_chrms = sample_data['chrom_chrms']

    pos_negs = [1,-1]

    all_continuous_samples = []
    num_cells = len(OEMs)

    for n in trange(num_cells):
        
        continuous_intra_samples = []
        
        for pn in pos_negs:

            pos_neg_intra_idx = torch.where((all_cell_labels[n] == pn) & (to_cuda(torch.tensor(all_cell_intra_inter[n])) > 0))[0]
            pos_neg_intra_chrms = to_cuda(torch.tensor(all_cell_chrms[n]))[pos_neg_intra_idx]

            pos_neg_intra_chrm_idx = torch.where(pos_neg_intra_chrms == chrn)[0]
            chrm_samples = all_cell_chrom_samples[n][pos_neg_intra_idx][pos_neg_intra_chrm_idx]

            adjusted_chrm_samples = np.zeros((len(chrm_samples),5))
            offset = int(all_cell_chrom_samples[n][pos_neg_intra_idx][pos_neg_intra_chrm_idx][0,-1])
            chrm_samples = chrm_samples[:,:2].cpu().numpy().astype(int) - offset

            chrlen = len(OEMs[n])

            chrm_linear_samples = chrm_samples[:,0] * chrlen + chrm_samples[:,1]
            # m_flattened = np.corrcoef(intra_matrices['chr%d' % i]).flatten()
            
            m_flattened = np.corrcoef(OEMs[n]).flatten()
            m_flattened[m_flattened > 0] = m_flattened[m_flattened > 0] / np.quantile(m_flattened[m_flattened > 0],0.975)
            m_flattened[m_flattened <= 0] = -(m_flattened[m_flattened <= 0] / np.quantile(m_flattened[m_flattened <= 0],0.025))
            m_flattened[m_flattened > 1] = 1
            m_flattened[m_flattened < -1] = -1
            c_flattened = m_flattened[chrm_linear_samples]
            
            # c_flattened = intra_matrices['chr%d' % i].flatten()[chrm_linear_samples]

            adjusted_chrm_samples[:,:2] = chrm_samples + offset
            adjusted_chrm_samples[:,2] = c_flattened
            adjusted_chrm_samples[:,3] = chrn
            adjusted_chrm_samples[:,4] = chrlen

            adjusted_del_idx = np.where(np.sign(c_flattened) != pn)[0]
            adjusted_chrm_samples = np.delete(adjusted_chrm_samples,adjusted_del_idx,axis=0)
            """if pn == -1:
                adjusted_chrm_samples[:,2] = 0"""
            
            continuous_intra_samples.append(adjusted_chrm_samples)

        continuous_intra_samples = to_cuda(torch.tensor(np.concatenate(continuous_intra_samples)).float())
        # continuous_intra_samples = torch.tensor(np.concatenate(continuous_intra_samples)).float()
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
    
    for chrn in range(chromosomes):
        
        sample_path = os.path.join(runtime_args['data_directory'],'chr%d_samples.pkl' % chrn)
        oe_path = os.path.join(runtime_args['data_directory'],'chr%d_oe.npy' % chrn)
        out_path = os.path.join(runtime_args['data_directory'],'chr%d_calibrated_samples.pkl' % chrn)

        print('Processing chromosome %d' % chrn)
        post_process_samples(
            sample_path,
            oe_path,
            out_path
        )
        
        torch.cuda.empty_cache()