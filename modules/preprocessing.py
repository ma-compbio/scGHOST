from cProfile import run
import numpy as np
import pickle
import gc
import os
import h5py

from utilities.chrom_sizes import chrom_sizes
from scipy.sparse import coo_matrix
from tqdm import trange

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

def parse_cell_types(runtime_args):
    label_info = pickle.load(open(runtime_args['label_info']['path'],'rb'))

    cell_type = runtime_args['cell_type']
    cell_type_filter = cell_type is not None
    cell_types = np.array(label_info[runtime_args['label_info']['cell_type_key']]).astype(str)
    cell_type_index = np.where(cell_types == cell_type) if cell_type_filter else np.arange(len(cell_types))

    return cell_type_index

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

def extract_OEMs(fname,cell_type_index,chrom_indices,chrom_num,chrom_start_end,save_path=None,offset=0,eps=1e-8):
    f = h5py.File(fname)
    
    chrom_size = chrom_start_end[chrom_num-1,1] - chrom_start_end[chrom_num-1,0]
    coords = np.array(f['coordinates'])

    num_cells = len(cell_type_index) # if num_cells is None else np.min([num_cells,len(cell_type_index)])
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
            os.path.join(runtime_args['schic_directory'],'chr%d_exp5_zinb_nbr_0_impute.hdf5' % (chrom_num)),
            cell_type_index,
            chrom_indices['chr%d' % chrom_num],
            chrom_num,
            chrom_start_end,
            save_path=os.path.join(runtime_args['data_directory'],'chr%d_oe' % (chrom_num)),
        )

        print('chr%d complete' % chrom_num)
        gc.collect()