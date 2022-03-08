#!/usr/bin/env python
'''
Compare candidate periods across N files and output one N-digit binary code per candidate.
In the binary code, "1" denotes detection  and "0" denotes non-detection.
ORDER MATTERS: Candidate detection in file i is denoted by "1" in the i^{th} position of the code (read from left to right).

Run using the following syntax.
python compare_cands.py -i <Configuration script of inputs> | tee <Log file>
'''
from __future__ import print_function
from __future__ import absolute_import
# Custom imports
from modules.general_utils import setup_logger_stdout, create_dir
from modules.read_config import read_config
# Standard packages
from argparse import ArgumentParser
from riptide.clustering import cluster1d
from tqdm import tqdm
import os, logging, time, sys, csv
import numpy as np
import pandas as pd
##############################################################
def myexecute(inputs_cfg):
    """
    Primary function that handles fake signal injection.

    Parameters
    ----------
    inputs_cfg : str
         configuration script of inputs
    """
    # Profile code execution.
    prog_start_time = time.time()

    # Read inputs from config file and set default parameter values, if applicable.
    hotpotato = read_config(inputs_cfg)
    hotpotato = set_defaults(hotpotato)
    logger = setup_logger_stdout() # Set logger output to stdout().

    # Read .csv files.
    N_files = len(hotpotato['csv_list']) # No. of .csv files
    logger.info('Total no. of input .csv files = %d'% (N_files))
    # Empty arrays to populate in subsequent lines
    file_index = np.array([], dtype=int) # File index of candidates
    chans = np.array([], dtype=int) # Channel indices of candidates
    radiofreqs = np.array([], dtype=np.float64) # Radio frequencies (MHz) of candidate detections
    bins = np.array([], dtype=int) # No. of bins across folded profile
    widths = np.array([], dtype=int) # No. of samples spanning width of optimal Boxcar filter
    periods = np.array([], dtype=np.float64) # Candidate periods (s)
    snrs = np.array([], dtype=np.float64) # Candidate S/N
    flags = np.array([]) # Harmonic flags assigned to candidates
    for i in range(N_files):
        df = pd.read_csv(hotpotato['CSV_DIR']+'/'+hotpotato['csv_list'][i], sep=',')
        logger.info('Reading file: %s'%  (hotpotato['csv_list'][i]))
        cand_chans = np.array(df['Channel'], dtype=int)
        cand_radiofreqs = np.array(df['Radio frequency (MHz)'], dtype=np.float64)
        cand_bins = np.array(df['Bins'], dtype=int)
        cand_widths = np.array(df['Best width'], dtype=int)
        cand_periods = np.array(df['Period (s)'], dtype=np.float64)
        cand_snrs = np.array(df['S/N'], dtype=np.float64)
        cand_flags = np.array(df['Harmonic flag'])
        N_cands = len(cand_chans)
        print('No. of candidates = %d \n'% (N_cands))
        # Append to grand arrays.
        file_index = np.append(file_index, np.ones(N_cands)*i)
        chans = np.append(chans, cand_chans)
        radiofreqs = np.append(radiofreqs, cand_radiofreqs)
        bins = np.append(bins, cand_bins)
        widths = np.append(widths, cand_widths)
        periods = np.append(periods, cand_periods)
        snrs = np.append(snrs, cand_snrs)
        flags = np.append(flags, cand_flags)
    # Consider only those candidates whose matched filtering S/N exceeds the user-specified threshold.
    chosen_cand_indices = np.where(snrs>=hotpotato['snr_cutoff'])[0]
    logger.info('Selecting candidates with S/N >= %.1f'% (hotpotato['snr_cutoff']))
    file_index = file_index[chosen_cand_indices]
    chans = chans[chosen_cand_indices]
    radiofreqs = radiofreqs[chosen_cand_indices]
    bins = bins[chosen_cand_indices]
    widths = widths[chosen_cand_indices]
    periods = periods[chosen_cand_indices]
    snrs = snrs[chosen_cand_indices]
    flags = flags[chosen_cand_indices]
    # Select only fundamental frequencies.
    logger.info('Working with fundamental frequencies only')
    f_idx = np.where(flags=='F')[0]
    file_index = file_index[f_idx]
    chans = chans[f_idx]
    radiofreqs = radiofreqs[f_idx]
    bins = bins[f_idx]
    widths = widths[f_idx]
    periods = periods[f_idx]
    snrs = snrs[f_idx]
    print('Final no. of candidates after pruning = %d \n'% (len(periods)))

    # Open output csvfile.
    create_dir(hotpotato['OUTPUT_DIR'])
    f = open(hotpotato['OUTPUT_DIR']+'/'+hotpotato['basename']+'_comparecands.csv', 'w')
    writer = csv.writer(f, delimiter=',')
    header = ['Channel', 'Radio frequency (MHz)', 'Bins', 'Best width', 'Period (s)', 'S/N', 'Code']
    writer.writerow(header) # Write header row.
    # Loop over channels.
    unique_chans = np.unique(chans)
    logger.info('Channel-wise grouping of candidate periods into clusters of radius %.2f ms'% (hotpotato['cluster_radius']*1.0e3))
    for ch in tqdm(unique_chans):
        # Channel selection
        ch_idx = np.where(chans==ch)[0]
        ch_files = file_index[ch_idx]
        ch_radiofreqs = radiofreqs[ch_idx]
        ch_bins = bins[ch_idx]
        ch_widths = widths[ch_idx]
        ch_periods = periods[ch_idx]
        ch_snrs = snrs[ch_idx]
        # Sort in order of increasing periods.
        sort_order = np.argsort(ch_periods)
        ch_files = ch_files[sort_order]
        ch_radiofreqs = ch_radiofreqs[sort_order]
        ch_bins = ch_bins[sort_order]
        ch_widths = ch_widths[sort_order]
        ch_periods = ch_periods[sort_order]
        ch_snrs = ch_snrs[sort_order]
        # Group candidates into distinct clusters in period space.
        cluster_indices = cluster1d(ch_periods, hotpotato['cluster_radius'], already_sorted=True)
        N_clusters = len(cluster_indices)
        for n in range(N_clusters):
            select_idx = cluster_indices[n][np.argmax(ch_snrs[cluster_indices[n]])]
            code = np.array(['0']*N_files) # Default code
            code[np.unique(ch_files[cluster_indices[n]]).astype(int)] = '1'
            code = ''.join(code)
            row = [ch, ch_radiofreqs[select_idx], ch_bins[select_idx], ch_widths[select_idx], ch_periods[select_idx], ch_snrs[select_idx], code]
            writer.writerow(row)
    # Close file cursor.
    f.close()

    # Calculate total run time for the code.
    prog_end_time = time.time()
    run_time = (prog_end_time - prog_start_time)/60.0
    logger.info('Code run time = %.3f minutes'% (run_time))
##############################################################
def set_defaults(hotpotato):
    """
    Set default values for keys in a dictionary of input parameters.

    Parameters
    ----------
    hotpotato : dictionary
         Dictionary of input parameters gathered from a configuration script

    Returns
    -------
    hotpotato : dictionary
        Input dictionary with keys set to default values
    """
    # Default S/N threshold = 7
    if hotpotato['snr_cutoff']=='':
        hotpotato['snr_cutoff'] = 7.0
    # Default cluster radius = 1 ms
    if hotpotato['cluster_radius']=='':
        hotpotato['cluster_radius'] = 1.0e-3
    return hotpotato
##############################################################
def main():
    """ Command line tool for running compare_cands.py """
    parser = ArgumentParser(description="Compare candidate periods across N files and output one N-digit binary code per candidate.")
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    required.add_argument('-i', action='store', required=True, dest='inputs_cfg', type=str,
                            help="Configuration script of inputs")
    parser._action_groups.append(optional)

    if len(sys.argv)==1:
        parser.print_help()
        sys.exit(1)

    parse_args = parser.parse_args()
    # Initialize parameter values
    inputs_cfg = parse_args.inputs_cfg

    # Run task using inputs from configuration script.
    myexecute(inputs_cfg)
##############################################################
if __name__=='__main__':
    main()
##############################################################
