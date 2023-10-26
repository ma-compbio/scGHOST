import os
import torch
import numpy as np
import pickle
import gc

from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import quantile_transform as sk_quantile_transform
from sklearn.cluster import KMeans
from utilities.parsers import parse_chrom_embeds, parse_chromosomes, parse_chromosomes, parse_higashi_scab, parse_cell_types
from utilities.helper import to_cuda
from sklearn.cluster import KMeans as SKMeans
from tqdm import trange, tqdm
from cuml import KMeans

DEFAULT_KMEANS_INIT = 10

def quantile_transform(data, n_quantiles=None):

    nq = n_quantiles if n_quantiles is not None else len(data)

    sorted_data, sort_id = torch.sort(data)
    data[sort_id] = torch.arange(data.shape[0],device=data.device,dtype=data.dtype)
    data = torch.floor(data / data.shape[0] * nq) / nq
    return data

def scghost_clustering(runtime_args):
    embed_data = parse_chrom_embeds(runtime_args)
    chromosomes = parse_chromosomes(runtime_args)
    cell_type_index = parse_cell_types(runtime_args)
    gpu_caching = False if 'cluster_gpu_caching' not in runtime_args else runtime_args['cluster_gpu_caching']

    kmeans_init = DEFAULT_KMEANS_INIT if 'kmeans_init' not in runtime_args else runtime_args['kmeans_init']

    chrom_embeds = embed_data['embeds']
    chrom_highlow = embed_data['highlow']

    N = embed_data['N']
    bar = trange(N) if cell_type_index is None else tqdm(cell_type_index)
    # bar = trange(N) if cell_type_index is None else tqdm(range(5))

    cell_labels = []
    cell_labels_transpose = []

    for cn in bar:
    # for cn in trange(25):
        
        inter_matrix = []
        
        for ii in range(0,len(chromosomes),2):
            chrom1 = chromosomes[ii]
            embed1 = chrom_embeds['{0}'.format(chrom1)][cn]
            embed1 = embed1 if gpu_caching else to_cuda(embed1)
            corr1 = torch.corrcoef(embed1)

            hi1 = chrom_highlow['{0}'.format(chrom1)]['high'][cn]
            lo1 = chrom_highlow['{0}'.format(chrom1)]['low'][cn]
            
            slc1 = corr1[hi1] - corr1[lo1]

            row = []
            
            for jj in range(1,len(chromosomes),2):
                chrom2 = chromosomes[jj]
                embed2 = chrom_embeds['{0}'.format(chrom2)][cn]
                embed2 = embed2 if gpu_caching else to_cuda(embed2)
                corr2 = torch.corrcoef(embed2)

                hi2 = chrom_highlow['{0}'.format(chrom2)]['high'][cn]
                lo2 = chrom_highlow['{0}'.format(chrom2)]['low'][cn]
            
                slc2 = corr2[hi2] - corr2[lo2]
                
                op = slc1.mean(dim=0)[:,None] * slc2.mean(dim=0)[None]

                opf = op.flatten()
                opf = quantile_transform(opf,n_quantiles=1000)
                opq = opf.reshape(op.shape)
                
                row.append(opq)
                
            row = torch.hstack(row)
            inter_matrix.append(row)
            
        # inter_matrix = torch.from_numpy(np.vstack(inter_matrix)).cuda()
        inter_matrix = torch.vstack(inter_matrix)
        
        L = KMeans(n_clusters=5,n_init=kmeans_init).fit_predict(inter_matrix)
        LT = KMeans(n_clusters=5,n_init=kmeans_init).fit_predict(inter_matrix.T)

        cell_labels.append(L.get())
        cell_labels_transpose.append(LT.get())
        # gc.collect()
        
    cell_labels = np.array(cell_labels)
    cell_labels_transpose = np.array(cell_labels_transpose)

    # align using hig_scab

    hig_scab,scAB_chrom,scAB_start = parse_higashi_scab(runtime_args)

    cmap = []
    rmap = []
    chrom_hig = {}
    cropped_indices = {}

    data_dir = runtime_args['data_directory']

    for ii in range(0,len(chromosomes),2):
        chrom = chromosomes[ii]
        
        chrom_indices = pickle.load(open(os.path.join(data_dir,'chrom_indices.pkl'),'rb'))['{0}'.format(chrom)]
        scab_chrom_indices = np.where(scAB_chrom == 'chr{0}'.format(runtime_args['chromosomes'][chrom]['integer']))[0]
        _,scab_crop,scghost_crop = np.intersect1d(scAB_start[scab_chrom_indices] // 500000,chrom_indices,return_indices=True)
        scab_indices = scab_chrom_indices[scab_crop]
        scghost_indices = chrom_indices[scghost_crop]
        cropped_indices['{0}'.format(chrom)] = scghost_indices
        
        rmap.append(
            np.vstack((
                np.ones(len(scghost_indices)) * runtime_args['chromosomes'][chrom]['integer'],
                np.arange(len(scghost_indices)),
                scghost_crop,
                scghost_indices
            )).T
        )
        
        chrom_hig['{0}'.format(chrom)] = hig_scab[:,scab_indices]
        
    for ii in range(1,len(chromosomes),2):
        chrom = chromosomes[ii]
        
        chrom_indices = pickle.load(open(os.path.join(data_dir,'chrom_indices.pkl'),'rb'))['{0}'.format(chrom)]
        scab_chrom_indices = np.where(scAB_chrom == 'chr{0}'.format(runtime_args['chromosomes'][chrom]['integer']))[0]

        _,scab_crop,scghost_crop = np.intersect1d(scAB_start[scab_chrom_indices] // 500000,chrom_indices,return_indices=True)
        scab_indices = scab_chrom_indices[scab_crop]
        scghost_indices = chrom_indices[scghost_crop]
        cropped_indices['{0}'.format(chrom)] = scghost_indices
        
        cmap.append(
            np.vstack((
                np.ones(len(scghost_indices)) * runtime_args['chromosomes'][chrom]['integer'],
                np.arange(len(scghost_indices)),
                scghost_crop,
                scghost_indices
            )).T
        )
        
        chrom_hig['{0}'.format(chrom)] = hig_scab[:,scab_indices]

    rmap = np.vstack(rmap)
    cmap = np.vstack(cmap)

    pickle.dump(cropped_indices,open(os.path.join(data_dir,'cropped_indices.pkl'),'wb'))

    chrom_sorted_labels = {}

    for chrom in chromosomes:
        chrom_sorted_labels['{0}'.format(chrom)] = []
        
        for i in bar:
            ab = chrom_hig['{0}'.format(chrom)][i]
            
            m = rmap if runtime_args['chromosomes'][chrom]['integer'] % 2 == 1 else cmap
            
            idx = np.where(m[:,0] == runtime_args['chromosomes'][chrom]['integer'])[0]
            lset = cell_labels if runtime_args['chromosomes'][chrom]['integer'] % 2 == 1 else cell_labels_transpose
            
            lbls = lset[i,idx]
            lbls_ab = np.zeros(5)
            
            for k in range(5):
                ii = np.where(lbls == k)[0]

                lbls_ab[k] = ab[ii].mean()
                
            lbls_order = lbls_ab.argsort()[::-1]
            lbls_sorted = lbls.copy()
            
            for k in range(5):
                lbls_sorted[lbls == lbls_order[k]] = k
            
            chrom_sorted_labels['{0}'.format(chrom)].append(lbls_sorted)
            
        chrom_sorted_labels['{0}'.format(chrom)] = np.array(chrom_sorted_labels['{0}'.format(chrom)])
        
    pickle.dump(chrom_sorted_labels,open(os.path.join(data_dir,'labels.pkl'),'wb'))