import json
import numpy as np
import torch
import h5py
import pickle
import gc
import os

from utilities.chrom_sizes import chrom_sizes
from utilities.helper import to_cuda
from sklearn.neighbors import NearestNeighbors

def parse_config(config_filepath):
    with open(config_filepath) as config_file:
        config_data = json.load(config_file)
        
        return config_data
        
def parse_higashi_scab(runtime_args):
    scAB = h5py.File(runtime_args['higashi_scab_path'])
    chromosomes = parse_chromosomes(runtime_args)

    scAB_chrom = np.array(scAB['compartment']['bin']['chrom']).astype(str)
    scAB_start = np.array(scAB['compartment']['bin']['start'])

    hig_scab = []

    num_cells = 0

    for i in range(len(scAB['compartment'])):
        if 'cell_%d' % i in scAB['compartment']:
            num_cells += 1

    for cn in range(num_cells):
        hig_scab.append(scAB['compartment']['cell_%d' % cn])

    hig_scab = np.array(hig_scab)

    return (hig_scab,scAB_chrom,scAB_start)

def parse_chrom_embeds(runtime_args,cuda=True):

    chromosomes = parse_chromosomes(runtime_args)
    gpu_caching = False if 'cluster_gpu_caching' not in runtime_args else runtime_args['cluster_gpu_caching']

    hig_scab,scAB_chrom,scAB_start = parse_higashi_scab(runtime_args)

    N = len(hig_scab)

    chrom_embeds = {}
    chrom_highlow = {}

    resolution = runtime_args['resolution']

    for chrom in chromosomes:
        
        ci_path = os.path.join(runtime_args['data_directory'],'chrom_indices.pkl') if runtime_args['chrom_indices'] is None else runtime_args['chrom_indices']
        chrom_indices = pickle.load(
            open(ci_path,'rb')
        )['{0}'.format(chrom)]

        scab_chrom_indices = np.where(scAB_chrom == 'chr{0}'.format(chrom))[0]
        _,scab_crop,scghost_crop = np.intersect1d(scAB_start[scab_chrom_indices] // resolution,chrom_indices,return_indices=True)
        scab_indices = scab_chrom_indices[scab_crop]
        scghost_indices = chrom_indices[scghost_crop]

        scab_highidx = np.argsort(hig_scab[:,scab_indices],axis=1)[:,-25:]
        scab_lowidx = np.argsort(hig_scab[:,scab_indices],axis=1)[:,:25]
        
        chrom_highlow['{0}'.format(chrom)] = {
            'high' : to_cuda(torch.tensor(scab_highidx)) if gpu_caching else torch.tensor(scab_highidx),
            'low' : to_cuda(torch.tensor(scab_lowidx)) if gpu_caching else torch.tensor(scab_lowidx)
        }
        
        embedding_flag = ('embeddings' in runtime_args['chromosomes'][chrom])

        embed_path = os.path.join(
            runtime_args['data_directory'],'{0}_embeddings.npy'.format(chrom)
        )
        if embedding_flag and runtime_args['chromosomes'][chrom]['embeddings'] is not None:
            embed_path = runtime_args['chromosomes'][chrom]['embeddings']

        scembeds = np.load(embed_path)
        scembeds = scembeds[:,scghost_crop]
        
        chrom_embeds['{0}'.format(chrom)] = to_cuda(torch.tensor(scembeds)) if gpu_caching else torch.tensor(scembeds)

        gc.collect()

    return {
        'embeds':chrom_embeds,
        'highlow':chrom_highlow,
        'N':N,
    }

def parse_chromosomes(runtime_args):

    sizes = chrom_sizes(runtime_args['chrom_sizes'])
    chromosomes = runtime_args['chromosomes']
    
    # deprecate this if condition
    if chromosomes == 'autosomes':
        chrom_list = []

        for chrom in sizes:
            chrom_num = chrom[3:]
            if chrom_num.isnumeric():
                chrom_list.append(int(chrom_num))
        chromosomes = np.array(chrom_list)
    else:
        chromosomes = np.array([c for c in chromosomes])
    
    return chromosomes

def parse_nearest_neighbors(runtime_args):

    cell_type = runtime_args['cell_type']
    label_info = pickle.load(open(runtime_args['label_info']['path'],'rb')) if runtime_args['label_info'] is not None else None

    embeddings = np.load(runtime_args['embeddings_path'])
    
    if label_info is not None and cell_type is not None:
        cell_type_key = runtime_args['label_info']['cell_type_key']
        cell_types = np.array(label_info[cell_type_key]).astype(str)
        cell_type_index = np.where(cell_types == cell_type)

        embeddings = embeddings[cell_type_index]

    nbrs = NearestNeighbors(n_neighbors=6).fit(embeddings)
    _,indices = nbrs.kneighbors(embeddings)

    return indices

def parse_cell_types(runtime_args):

    if runtime_args['label_info'] is None:
        return
    
    label_info = pickle.load(open(runtime_args['label_info']['path'],'rb'))

    cell_type = runtime_args['cell_type']

    if cell_type is None:
        return
    
    cell_type_filter = cell_type is not None
    cell_types = np.array(label_info[runtime_args['label_info']['cell_type_key']]).astype(str)
    cell_type_index = np.where(cell_types == cell_type)[0] if cell_type_filter else np.arange(len(cell_types))

    return cell_type_index