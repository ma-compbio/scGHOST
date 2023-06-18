import json
import numpy as np
import h5py
import pickle
import gc
import os

from utilities.chrom_sizes import chrom_sizes
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

    for cn in range(len(scAB['compartment'])-3):
        hig_scab.append(scAB['compartment']['cell_%d' % cn])

    hig_scab = np.array(hig_scab)

    return (hig_scab,scAB_chrom,scAB_start)

def parse_chrom_embeds(runtime_args):

    chromosomes = parse_chromosomes(runtime_args)

    hig_scab,scAB_chrom,scAB_start = parse_higashi_scab(runtime_args)

    N = len(hig_scab)

    chrom_embeds = {}
    chrom_highlow = {}

    for chrom in chromosomes:
        
        chrom_indices = pickle.load(open(os.path.join(runtime_args['data_directory'],'chrom_indices.pkl'),'rb'))['chr%d' % chrom]
        scab_chrom_indices = np.where(scAB_chrom == 'chr%d' % chrom)[0]
        _,scab_crop,scghost_crop = np.intersect1d(scAB_start[scab_chrom_indices] // 500000,chrom_indices,return_indices=True)
        scab_indices = scab_chrom_indices[scab_crop]
        scghost_indices = chrom_indices[scghost_crop]

        scab_highidx = np.argsort(hig_scab[:,scab_indices],axis=1)[:,-25:]
        scab_lowidx = np.argsort(hig_scab[:,scab_indices],axis=1)[:,:25]
        
        chrom_highlow['chr%d' % chrom] = {'high' : scab_highidx,'low' : scab_lowidx}
        
        scembeds = np.load(os.path.join(runtime_args['data_directory'],'chr%d_embeddings.npy' % chrom))
        scembeds = scembeds[:,scghost_crop]
        
        chrom_embeds['chr%d' % chrom] = scembeds

        gc.collect()

    return {
        'embeds':chrom_embeds,
        'highlow':chrom_highlow,
        'N':N,
    }

def parse_chromosomes(runtime_args):

    sizes = chrom_sizes(runtime_args['chrom_sizes'])
    chromosomes = runtime_args['chromosomes']
    
    if chromosomes == 'autosomes':
        chrom_list = []

        for chrom in sizes:
            chrom_num = chrom[3:]
            if chrom_num.isnumeric():
                chrom_list.append(int(chrom_num))
        chromosomes = np.array(chrom_list)
    else:
        chromosomes = np.array(chromosomes)

    return chromosomes

def parse_nearest_neighbors(runtime_args):

    cell_type = runtime_args['cell_type']
    label_info = pickle.load(open(runtime_args['label_info']['path'],'rb'))
    cell_type_key = runtime_args['label_info']['cell_type_key']
    cell_types = np.array(label_info[cell_type_key]).astype(str)
    cell_type_index = np.where(cell_types == cell_type)

    cell_type_embeddings = np.load(runtime_args['embeddings_path'])[cell_type_index]

    nbrs = NearestNeighbors(n_neighbors=6).fit(cell_type_embeddings)
    _,indices = nbrs.kneighbors(cell_type_embeddings)

    nearest_neighbors = indices
    
    return nearest_neighbors

def parse_cell_types(runtime_args):
    label_info = pickle.load(open(runtime_args['label_info']['path'],'rb'))

    cell_type = runtime_args['cell_type']
    cell_type_filter = cell_type is not None
    cell_types = np.array(label_info[runtime_args['label_info']['cell_type_key']]).astype(str)
    cell_type_index = np.where(cell_types == cell_type) if cell_type_filter else np.arange(len(cell_types))

    return cell_type_index
