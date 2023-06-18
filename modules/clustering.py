import os
import numpy as np
import pickle
import gc
import matplotlib.pyplot as plt

from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import quantile_transform
from sklearn.cluster import KMeans
from scipy.stats import zscore
from utilities.parsers import parse_chrom_embeds, parse_chromosomes, parse_chromosomes, parse_higashi_scab
from sklearn.cluster import KMeans
from tqdm import trange

def scghost_clustering(runtime_args):
    embed_data = parse_chrom_embeds(runtime_args)
    chromosomes = parse_chromosomes(runtime_args)

    chrom_embeds = embed_data['embeds']
    chrom_highlow = embed_data['highlow']
    N = embed_data['N']

    cell_labels = []
    cell_labels_transpose = []

    for cn in trange(N):
        
        inter_matrix = []
        
        for ii in range(0,len(chromosomes),2):
            chrom1 = chromosomes[ii]
            embed1 = chrom_embeds['chr%d' % chrom1][cn]
            corr1 = np.corrcoef(embed1)

            hi1 = chrom_highlow['chr%d' % chrom1]['high'][cn]
            lo1 = chrom_highlow['chr%d' % chrom1]['low'][cn]
            
            slc1 = corr1[hi1] - corr1[lo1]
            
            row = []
            
            for jj in range(1,len(chromosomes),2):
                chrom2 = chromosomes[jj]
                embed2 = chrom_embeds['chr%d' % chrom2][cn]
                corr2 = np.corrcoef(embed2)

                hi2 = chrom_highlow['chr%d' % chrom2]['high'][cn]
                lo2 = chrom_highlow['chr%d' % chrom2]['low'][cn]
            
                slc2 = corr2[hi2] - corr2[lo2]
                
                op = slc1.mean(axis=0)[:,None] * slc2.mean(axis=0)[None]
                
                opf = op.flatten()
                opf = quantile_transform(opf[:,None]).flatten()
                opq = opf.reshape(op.shape)
                
                row.append(opq)
                
            row = np.hstack(row)
            inter_matrix.append(row)
            
        inter_matrix = np.vstack(inter_matrix)
        
        L = KMeans(n_clusters=5).fit_predict(inter_matrix)
        LT = KMeans(n_clusters=5).fit_predict(inter_matrix.T)
        
        cell_labels.append(L)
        cell_labels_transpose.append(LT)
        gc.collect()
        
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
        
        chrom_indices = pickle.load(open(os.join(data_dir,'chrom_indices.pkl'),'rb'))['chr%d' % chrom]
        scab_chrom_indices = np.where(scAB_chrom == 'chr%d' % chrom)[0]
        _,scab_crop,scghost_crop = np.intersect1d(scAB_start[scab_chrom_indices] // 500000,chrom_indices,return_indices=True)
        scab_indices = scab_chrom_indices[scab_crop]
        scghost_indices = chrom_indices[scghost_crop]
        cropped_indices['chr%d' % chrom] = scghost_indices
        
        rmap.append(
            np.vstack((
                np.ones(len(scghost_indices)) * chrom,
                np.arange(len(scghost_indices)),
                scghost_crop,
                scghost_indices
            )).T
        )
        
        chrom_hig['chr%d' % chrom] = hig_scab[:,scab_indices]
        
    for ii in range(1,len(chromosomes),2):
        chrom = chromosomes[ii]
        
        chrom_indices = pickle.load(open(os.join(data_dir,'chrom_indices.pkl'),'rb'))['chr%d' % chrom]
        scab_chrom_indices = np.where(scAB_chrom == 'chr%d' % chrom)[0]
        _,scab_crop,scghost_crop = np.intersect1d(scAB_start[scab_chrom_indices] // 500000,chrom_indices,return_indices=True)
        scab_indices = scab_chrom_indices[scab_crop]
        scghost_indices = chrom_indices[scghost_crop]
        cropped_indices['chr%d' % chrom] = scghost_indices
        
        cmap.append(
            np.vstack((
                np.ones(len(scghost_indices)) * chrom,
                np.arange(len(scghost_indices)),
                scghost_crop,
                scghost_indices
            )).T
        )
        
        chrom_hig['chr%d' % chrom] = hig_scab[:,scab_indices]

    rmap = np.vstack(rmap)
    cmap = np.vstack(cmap)

    pickle.dump(cropped_indices,open(os.join(data_dir,'cropped_indices.pkl'),'wb'))

    chrom_sorted_labels = {}

    for chrom in chromosomes:
        chrom_sorted_labels['chr%d' % chrom] = []
        
        for i in trange(N):
            ab = chrom_hig['chr%d' % chrom][i]
            
            m = rmap if chrom % 2 == 1 else cmap
            
            idx = np.where(m[:,0] == chrom)[0]
            lset = cell_labels if chrom % 2 == 1 else cell_labels_transpose
            
            lbls = lset[i,idx]
            lbls_ab = np.zeros(5)
            
            for k in range(5):
                ii = np.where(lbls == k)[0]

                lbls_ab[k] = ab[ii].mean()
                
            lbls_order = lbls_ab.argsort()[::-1]
            lbls_sorted = lbls.copy()
            
            for k in range(5):
                lbls_sorted[lbls == lbls_order[k]] = k
            
            chrom_sorted_labels['chr%d' % chrom].append(lbls_sorted)
            
        chrom_sorted_labels['chr%d' % chrom] = np.array(chrom_sorted_labels['chr%d' % chrom])
        
    pickle.dump(chrom_sorted_labels,open(os.join(data_dir,'labels.pkl'),'wb'))

"""
Legacy K-Means clustering per chromosome
"""
def kmeans_clustering(runtime_args):
    nc = runtime_args['num_clusters']
    chromosomes = parse_chromosomes(runtime_args)

    clusters = {}

    for chrom in chromosomes:
        stacked_pcs = []
        stacked_per_cell_labels = []

        chrom_indices = pickle.load(open(
            os.path.join(runtime_args['data_directory'],'chrom_indices.pkl'),'rb'
        ))['chr%d' % chrom]

        embeddings = np.load(
            os.path.join(runtime_args['data_directory'],'chr%d_embeddings.npy')
        )
        
        num_cells = len(embeddings)
        embedding_corrs = np.zeros((embeddings.shape[0],embeddings.shape[1],embeddings.shape[1]))

        for i in range(num_cells):
            embedding_corrs[i] = np.corrcoef(embeddings[i])

        stacked_corrs = embedding_corrs.reshape(embedding_corrs.shape[0]*embedding_corrs.shape[1],embedding_corrs.shape[2])

        nc = 5
        L = KMeans(n_clusters=nc).fit_predict(stacked_corrs)

        clusters['chr%d' % chrom] = L

    out_path = os.path.join(runtime_args['data_directory'],'clusters.pkl')
    pickle.dump(clusters,open(out_path,'wb'))
