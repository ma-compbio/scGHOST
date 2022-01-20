# README for scGHOST
Please email kxiong@andrew.cmu.edu with any questions about installation or usage.

# Installation

To install scGHOST, clone the repository and install the necessary requirements by running the following command on a Linux terminal:

`pip install -r requirements.txt`

Systems without a CUDA-capable GPU can also install scGHOST using the same dependencies.

We recommend using Anaconda to create a python 3.7 environment using

`conda install --name scghost python=3.7`

## Requirements

All Python dependencies can be installed by running

`pip install -r requirements.txt` (see Installation)

### Python dependencies:

* Python (3.7.X). Any version of Python 3.7 should work, but the code has been confirmed to work on 3.7.11.
* PyTorch (1.10.1)
* h5py (2.8.0)
* numpy (1.15.4)
* scipy (1.11.0)

# Usage

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
- `cell_type` : the cell type in the dataset to apply scGHOST on; use `null` to apply scGHOST to all cell types in the dataset. default: `null`