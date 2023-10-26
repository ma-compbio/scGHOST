import argparse
import os
import pickle
import torch
import numpy as np
import gc

from utilities.parsers import parse_config, parse_chromosomes, parse_cell_types, parse_nearest_neighbors

from modules.preprocessing import compute_chrom_indices, extract_OEMs
from modules.postprocessing import post_process_samples
from modules.random_walk import sample_chrom
from modules.embedding import embed_single_cells_unified, prep_pairs_labels
from modules.clustering import scghost_clustering
from tqdm import trange

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', metavar='cfg', type=str,default='./config.json',help='Path to the configuration file')

    args = parser.parse_args()

    runtime_args = parse_config(args.config)

    os.makedirs(runtime_args['data_directory'],exist_ok=True)

    num_walks = runtime_args['random_walk']['num_walks']
    neighbor_contacts = True if 'neighbor_contacts' not in runtime_args else runtime_args['neighbor_contacts']
    gpu_uniques = False if 'gpu_uniques' not in runtime_args else runtime_args['gpu_uniques']

    # define globals
    print('Parsing chromosomes')
    chromosomes = parse_chromosomes(runtime_args)
    
    print('Parsing chromosome indices')
    chrom_indices = compute_chrom_indices(runtime_args) if runtime_args[
        'chrom_indices'
    ] is None else pickle.load(open(runtime_args['chrom_indices'],'rb'))
    
    print('Parsing cell types')
    cell_type = runtime_args['cell_type']
    cell_type_index = parse_cell_types(runtime_args)

    print('Parsing remaining global variables')
    chrom_start_end = np.load(os.path.join(runtime_args['schic_directory'],'chrom_start_end.npy'))

    nearest_neighbors = None
        
    if 'nearest_neighbor_override' in runtime_args and runtime_args['nearest_neighbor_override'] is not None:
        print('Using nearest neighbor override')
        nearest_neighbors = np.load(runtime_args['nearest_neighbor_override']) if runtime_args['nearest_neighbor_override'] is not None else parse_nearest_neighbors(runtime_args)
    else:
        nearest_neighbors = parse_nearest_neighbors(runtime_args)
    
    batch_size = runtime_args['batch_size']
    n_epochs = runtime_args['epochs']

    # per chromosome loop
    for chrom in chromosomes:

        # if embedding already generated, skip
        if os.path.exists(
            os.path.join(runtime_args['data_directory'],'{0}_embeddings.npy'.format(chrom))
        ):
            continue

        print('Processing chromosome {0}'.format(chrom))
        impute_path = runtime_args['chromosomes'][chrom]['imputed']        

        # compute O/E matrices

        oem_override = None if 'oe_matrices' not in runtime_args['chromosomes'][chrom] else runtime_args['chromosomes'][chrom]['oe_matrices']

        OEMs = extract_OEMs(
            os.path.join(runtime_args['schic_directory'],impute_path),
            cell_type_index,
            chrom_indices[chrom],
            None,
            runtime_args['chromosomes'][chrom]['integer'],
            chrom_start_end,
            save_path=None,
            eps=runtime_args['eps']
        ) if oem_override is None else np.load(oem_override)['contact_maps']
        gc.collect()

        # random walk
        OEMs = torch.tensor(OEMs)
        corr_OEMs = torch.zeros_like(OEMs)

        for i in trange(len(OEMs)):
            corr_OEMs[i] = torch.nan_to_num(torch.corrcoef(OEMs[i]))
            corr_OEMs[i].fill_diagonal_(0)

        corr_OEMs = corr_OEMs.type(torch.bfloat16)

        gc.collect()
        
        all_cell_chrom_samples = sample_chrom(chrom,corr_OEMs,np.arange(len(corr_OEMs)),nearest_neighbors,num_walks=num_walks)

        del corr_OEMs
        gc.collect()
        torch.cuda.empty_cache()

        # label calibration
        all_continuous_pairs,all_continuous_labels = post_process_samples(
            all_cell_chrom_samples,
            OEMs,
            nearest_neighbors,
            neighbor_contacts=neighbor_contacts
        )

        all_continuous_pairs,all_continuous_labels = prep_pairs_labels(
            all_continuous_pairs,
            all_continuous_labels,
            OEMs[0].shape[0],
            np.arange(len(OEMs))
        )
        
        del all_cell_chrom_samples
        gc.collect()
        torch.cuda.empty_cache()

        # embedding
        output_file = os.path.join(runtime_args['data_directory'], '{0}_embeddings'.format(chrom))

        embed_single_cells_unified(
            all_continuous_pairs,
            all_continuous_labels,
            OEMs,
            output_file,
            epochs=n_epochs,
            cell_nums=None,
            batch_size=batch_size,
            verbose=True,
            prepped=True
        )

        del all_continuous_labels,all_continuous_pairs

        gc.collect()
        torch.cuda.empty_cache()

    # cluster on all embeddings
    print('Clustering')
    scghost_clustering(runtime_args)