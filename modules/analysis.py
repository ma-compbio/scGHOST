# sort chromosome subcompartments using single cell AB compartments from Higashi
import h5py
import os
import seaborn as sns

os.environ["OMP_NUM_THREADS"] = "10"

import numpy as np
from tqdm import trange, tqdm
import pickle
import pandas as pd
import argparse
from umap import UMAP
from fbpca import pca
from sklearn.preprocessing import StandardScaler, quantile_transform
from sklearn.decomposition import PCA
from scipy.stats import rankdata
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
import json

def get_expected(M,eps=1e-8):
    E = np.zeros_like(M)
    l = len(M)

    for i in range(M.shape[0]):
        contacts = np.diag(M,i)
        expected = contacts.sum() / (l-i)
        # expected = np.median(contacts)
        x_diag,y_diag = np.diag_indices(M.shape[0]-i)
        x,y = x_diag,y_diag+i
        E[x,y] = expected

    E += E.T
    E = np.nan_to_num(E) + eps
    
    return E
    
def get_oe_matrix(M):
    E = get_expected(M)
    oe = np.nan_to_num(M / E)
    np.fill_diagonal(oe,1)
    
    return oe

# tailored for k=5 and pfc
def prep_scatterplot(embeddings_dir,chrom_indices_file,scAB_file,output_file='tutorial_embeds.hdf5'):

    chrom_indices = pickle.load(open(chrom_indices_file,'rb'))
    stacked_pcs = []

    for chrom_num in range(1,23):
        chrom_indices = pickle.load(open('/mnt/e/data/scghost_pfc_output/chrom_indices.pkl','rb'))['%d' % chrom_num]

        sparse_M = np.load('/mnt/e/data/pfc/chr%d_sparse_adj.npy' % chrom_num,allow_pickle=True)
        pseudo_bulk = sparse_M.sum(axis=0).toarray()
        cov = np.sqrt(pseudo_bulk.sum(axis=1))
        pseudo_bulk /= cov[None,:]
        pseudo_bulk /= cov[:,None]
        pseudo_bulk = np.nan_to_num(pseudo_bulk)[chrom_indices][:,chrom_indices]
        pseudo_OE = get_oe_matrix(pseudo_bulk)

        Rpool = np.nan_to_num(np.corrcoef(pseudo_OE))
        Rpoolmean = Rpool.mean(axis=0,keepdims=True)
        Rpool = Rpool - Rpoolmean
        _,_,V = np.linalg.svd(Rpool)

        Es = np.load(os.path.join(embeddings_dir,f'/mnt/e/data/scghost_pfc_output/{chrom_num}_embeddings.npy'))
        embedding_corrs = np.zeros((Es.shape[0],Es.shape[1],Es.shape[1]))

        num_cells = len(Es)

        for i in trange(num_cells):
            embedding_corrs[i] = np.corrcoef(Es[i])

        pcs = np.zeros((Es.shape[0],Es.shape[1]))

        for i,ec in enumerate(embedding_corrs):
            tec = ec - Rpoolmean
            pc = tec.dot(V[0,:].T)
            pcs[i] = pc
            
        stacked_pcs.append(pcs)
        
    stacked_pcs = np.hstack(stacked_pcs)

    with h5py.File(output_file,'w') as f:
        f.create_group('compartment')
        f['compartment'].create_group('bin')

        for i in range(num_cells):
            f['compartment'].create_dataset('cell_%d' % i,data=stacked_pcs[i])


def get_config(config_path = "./config.jSON"):
    c = open(config_path,"r")
    return json.load(c)


def parse_args():
    parser = argparse.ArgumentParser(description="Higashi single cell compartment calling")
    parser.add_argument('-c', '--config', type=str, default="./config.JSON")
    
    return parser.parse_args()


def get_palette(label_order, label_name=None, config=None):
    try:
        palette = config['vis_palette'][label_name]
    except:
        pal1 = list(sns.color_palette("Paired"))
        pal2 = list(sns.color_palette("Set2"))
        pal3 = list(sns.color_palette("husl", 12))
        # pal = pal1 + pal2 + pal3 + pal1
        # pal = pal1 + pal3 + pal2
        pal_all = pal1 + pal2 + pal3 + pal1 + pal2 + pal3
        if len(label_order) <= 10:
            palette = list([f'C{_}' for _ in range(len(label_order))])
        else:
            palette = pal_all[:len(label_order)]
    return palette


def sc_compartment2embedding(embeds_path,data_dir,output_file="tutorial_scatterplot.pdf",extra="", save_name=""):
    label_info = pickle.load(open(os.path.join(data_dir, "label_info.pickle"), "rb"))
    label = np.array(label_info["cluster label"])
    print(label)
    
    ids = np.arange(4238)
    label = label[ids]
    total_feats = []
    
    with h5py.File(embeds_path, "r") as cp_f:
        print(cp_f.keys())
        cp = cp_f['compartment']
        
        for id_ in trange(len(label)):
            v = np.array(cp['cell_%d' % id_])
            total_feats.append(v)
    
    feats = np.stack(total_feats, axis=0)
    print(feats.shape)
    
    pal = get_palette(np.unique(label))
    
    pal = {'L2/3': '#e51f4e', 'L4': '#45af4b', 'L5': '#ffe011', 'L6': '#0081cc',
           'Ndnf': '#ff7f35', 'Vip': '#951eb7', 'Pvalb': '#4febee',
           'Sst': '#ed37d9', 'Astro': '#d1f33c', 'ODC': '#f9bdbb',
           'OPC': '#067d81', 'MG': '#e4bcfc', 'MP': '#ab6c1e',
           "Endo": '#780100'}
    

    
    temp = quantile_transform(feats, output_distribution='uniform', n_quantiles=int(1.0 * feats.shape[0]))
    print(feats.shape)
    size = 32
    pca = PCA(n_components=size)
    temp = pca.fit_transform(temp)

    vec = UMAP(n_components=2).fit_transform(temp)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(x=vec[:, 0], y=vec[:, 1], hue=label, linewidth=0, s=2, alpha=1.0, palette=pal)
    #
    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    ax.legend(handles=handles, labels=labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close('all')