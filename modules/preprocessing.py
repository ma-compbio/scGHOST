from cProfile import run
import numpy as np
import pickle
import gc
import os
import h5py

from utilities.parsers import parse_chromosomes, parse_cell_types
from utilities.chrom_sizes import chrom_sizes
from scipy.sparse import coo_matrix
from tqdm import trange

def compute_chrom_indices(runtime_args):

    chromosomes = parse_chromosomes(runtime_args)
    cell_type_index = parse_cell_types(runtime_args)

    chrom_indices = {}

    for n in trange(len(chromosomes)):

        chrom = chromosomes[n]
        
        sparse_M = np.load(os.path.join(runtime_args['schic_directory'],'chr%d_sparse_adj.npy' % chrom),allow_pickle=True)
        sparse_M = sparse_M[cell_type_index]

        M = sparse_M.sum(axis=0).toarray()

        nongap = np.where(np.sum(M > 0, axis=-1, keepdims=False) >= (0.1 * M.shape[0]))[0]
        
        chrom_indices['chr%d' % chrom] = nongap
        
    gc.collect()

    data_dir = runtime_args['data_directory']
    pickle.dump(chrom_indices,open(
            os.path.join(data_dir,'chrom_indices.pkl'),'wb'
    ))

def extract_OEMs(fname,cell_type_index,chrom_indices,num_cells,chrom_num,chrom_start_end,save_path=None,eps=1e-8):
    f = h5py.File(fname)
    
    chrom_size = chrom_start_end[chrom_num-1,1] - chrom_start_end[chrom_num-1,0]
    coords = np.array(f['coordinates'])

    num_cells = len(cell_type_index) if num_cells is None else np.min([num_cells,len(cell_type_index)])
    cells_data = np.array([np.array(f['cell_%d' % cell_type_index[i]]) for i in range(num_cells)])

    OEMs = []
    Ms = []

    for cell_num in trange(num_cells):
        M = coo_matrix((cells_data[cell_num],(coords[:,0],coords[:,1])),shape=(chrom_size,chrom_size)).toarray()
        M += M.T

        # construct expected matrix
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

        OE = M / E
        OE = OE[chrom_indices][:,chrom_indices]
        OE[OE == 0] = 1
        OE = np.log(OE)
        Ms.append(M[chrom_indices][:,chrom_indices])
        OEMs.append(OE)

    OEMs = np.array(OEMs)
    Ms = np.array(Ms)

    # print(OEMs.shape)
    if save_path is None:
        return OEMs, Ms
    else:
        # np.savez_compressed(save_path,oe=OEMs,observed=Ms)
        np.save(save_path,OEMs)
        np.save(save_path+'_observed',Ms)

def compute_observed_over_expected(runtime_args):
    
    chrom_start_end = np.load(os.path.join(runtime_args['schic_directory'],'chrom_start_end.npy'))
    cell_type = runtime_args['cell_type']
    chrom_indices = pickle.load(open(os.path.join(runtime_args['data_directory'],'chrom_indices.pkl'),'rb'))
    chromosomes = parse_chromosomes(runtime_args)

    for n in range(len(chromosomes)):
        
        chrom_num = chromosomes[n]
        cell_type_index = parse_cell_types(runtime_args)

        extract_OEMs(
            os.path.join(runtime_args['schic_directory'],'chr%d_exp1_nbr_5_impute.hdf5' % (chrom_num)),
            cell_type_index,
            chrom_indices['chr%d' % chrom_num],
            None,
            chrom_num,
            chrom_start_end,
            save_path=os.path.join(runtime_args['data_directory'],'chr%d_oe' % (chrom_num)),
            eps=runtime_args['eps']
        )

        print('chr%d complete' % chrom_num)
        gc.collect()
