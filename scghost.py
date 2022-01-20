import argparse
from cProfile import run
from utilities.parsers import parse_config

from modules.preprocessing import compute_chrom_indices, compute_observed_over_expected

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', metavar='cfg', type=str,default='./config.json',help='Path to the configuration file')

    args = parser.parse_args()

    runtime_args = parse_config(args.config)

    # compute_chrom_indices(runtime_args)
    compute_observed_over_expected(runtime_args)
    print(runtime_args)