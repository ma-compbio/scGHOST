import os
import numpy as np
import pickle
import matplotlib.pyplot as plt

from modules.preprocessing import parse_chromosomes
from sklearn.cluster import KMeans
from tqdm import trange

def kmeans_clustering(runtime_args):
    nc = runtime_args['num_clusters']
    chromosomes = parse_chromosomes(runtime_args['chromosomes'])

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