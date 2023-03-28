# Overview of scGHOST

![Overview of scGHOST](scghost_overview.png)

scGHOST is an unsupervised single-cell subcompartment annotation method based on graph embedding with constrained random walk sampling.
scGHOST is designed to be run on a single-cell Hi-C (scHi-C) dataset which has undergone imputation by [Higashi](https://github.com/ma-compbio/Higashi) ([Zhang et al. 2022](https://www.nature.com/articles/s41587-021-01034-y)).
scGHOST assigns embeddings to genomic loci in the genomes of individual cells by viewing scHi-C as graphs whose vertices are genomic loci and edges are the contact frequencies among loci.
While scGHOST is developed for scHi-C data, it can also identify single-cell subcompartments in single-cell genome imaging data.

# Running scGHOST

## Installation

Before installing any Python packages, we strongly recommend using Anaconda (please refer to the [Anaconda](https://anaconda.org/) webpage for `conda` installation instructions) to create a python 3.7 environment using the following command:

`conda install --name scghost python=3.7`

After creating the environment, activate it using:

`conda activate scghost`

scGHOST requires the following Python packages:
* PyTorch (1.10.1)
* scikit-learn
* h5py
* tqdm
* mkl-fft,mkl-random,mkl-service
* numpy
* scipy

Users can install scGHOST dependencies using `pip` by cloning this repository and installing the necessary requirements using the following command:

`pip install -r requirements.txt`

Systems without a CUDA-capable GPU can also install scGHOST using the same dependencies, but note that runtimes on a CPU-only system may be much longer than on a GPU-enabled system.

## Hardware Requirements

scGHOST uses up to 60 GB of memory for a single-cell dataset of approximately 4000 cells.
We therefore recommend a machine with at least 64 GB of memory to avoid sluggish performance or memory errors at runtime.

## Usage

scGHOST can be run using the following command:

`python scghost.py --config <configuration_filepath.json>`

`configuration_filepath` is the filepath to a custom configuration file adhering to the JSON format. By default, scGHOST uses the included config.json file, which can be modified to the user's specifications.

## Configuration file

- `schic_directory` : the directory containing Higashi-imputed single-cell Hi-C maps.
- `label_info` : `label_info.pickle` file following the [format in Higashi](https://github.com/ma-compbio/Higashi/wiki/Input-Files).
  - `path` : the file path of the `label_info.pickle` file
  - `cell_type_key` : the key in `label_info.pickle` with a list of the cell types in the dataset
- `data_directory` : the output directory of scGHOST
- `chromosomes` : the list of chromosomes to apply scGHOST to. default: autosomes
- `chrom_sizes` : file path to the chromosome sizes file. default: `data/hg38.chrom.sizes`
- `embeddings_path` : file path to the Higashi embeddings `.npy` file for each cell in the dataset
- `higashi_scab_path` : file path to Higashi scA/B scores `.h5` file
- `cell_type` : the cell type in the dataset to apply scGHOST on; use `null` to apply scGHOST to all cell types in the dataset. default: `null`
- `random_walk` : random walk parameters
  - `num_walks` : number of random walks per iteration. default: 50
  - `ignore_top` : the top and bottom percentile to be ignored, to remove extreme values in the input matrix. default: 0.02
  - `top_percentile` : the top percentiles within which random walks are performed. default: 0.25
- `eps` : small float value to prevent dividing by zero in some functions. default: 1e-8
- `num_clusters` : number of clusters to partition chromosomes into

## Contact
Please email kxiong@andrew.cmu.edu or raise an issue in the github repository with any questions about installation or usage.