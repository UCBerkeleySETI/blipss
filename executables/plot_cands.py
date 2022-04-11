#!/usr/bin/env python
'''
Plot candidate verification plots including periodograms, average pulse profiles and phase-time diagrams.

Run using the following syntax.
python plot_cands.py -i <Configuration script of inputs> | tee <Log file>
'''
from __future__ import print_function
from __future__ import absolute_import
# Custom imports
from modules.general_utils import setup_logger_stdout, create_dir
from modules.read_config import read_config
from modules.read_data import read_watfile
from modules.plotting import candverf_plot
# Standard packages
from argparse import ArgumentParser
from riptide import TimeSeries, ffa_search
import os, logging, time, sys
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

    # Read .csv file of periodicity candidates.
    logger.info('Reading file: %s'% (hotpotato['csvfile']))
    df = pd.read_csv(hotpotato['csvfile'], dtype={'Code':'string'})
    cand_codes = np.array(df['Code'])
    # Find indices of candidates to plot.
    idx = np.array([])
    for code in hotpotato['codes_plot']:
        idx = np.append(idx, np.where(cand_codes==code)[0])
    idx = np.array(idx, dtype=int)
    N_cands = len(idx)
    print('No. of candidates selected for plotting = %d\n'% (N_cands))
    # Properties of chosen candidates
    chosencand_chans = np.array(df['Channel'], dtype=int)[idx]
    chosencand_periods = np.array(df['Period (s)'], dtype=np.float64)[idx]
    chosencand_bins = np.array(df['Bins'], dtype=int)[idx]
    chosencand_codes = cand_codes[idx]

    # Read data.
    N_datafiles = len(hotpotato['datafile_list'])
    start_mjds = [] # Store start MJDs (UTC) of data sets
    all_data = [] # Store 2D dynamic spectra arrays.
    for i in range(N_datafiles):
        logger.info('Reading data from %s'% (hotpotato['datafile_list'][i]))
        wat = read_watfile(hotpotato['DATA_DIR'] + '/' + hotpotato['datafile_list'][i], hotpotato['mem_load'])
        data = wat.data[:,0,:].T
        if wat.header['foff']<0:
            data = np.flip(data,axis=0)
        # Store data arrays.
        all_data.append(data)
        # Store start MJDs.
        start_mjds.append(wat.header['tstart'])

    create_dir(hotpotato['PLOT_DIR'])
    # Produce candidate plots one by one.
    for i in range(N_cands):
        chan = chosencand_chans[i]
        period = chosencand_periods[i]
        bins = chosencand_bins[i]
        code = chosencand_codes[i]
        logger.info('Working with candidate %d'% (i+1))
        print('Channel = %d'% (chan))
        print('Period = %s s'% (period))
        print('Bins = %d'% (bins))
        print('Code = %s'% (code))

        # Compute periodogram of relevant time series from each data file.
        periodograms = [] # Store periodograms from different data files.
        detrended_ts = [] # Store detrended time series from different data files.
        max_snrs = [] # Max S/N in periodograms from each data file
        for j in range(N_datafiles):
            raw_ts = TimeSeries.from_numpy_array(all_data[j][chan], tsamp = wat.header['tsamp'])
            dts, pgram = ffa_search(raw_ts, period_min=hotpotato['min_period'], period_max=hotpotato['max_period'],
                                    fpmin=hotpotato['fpmin'], bins_min=hotpotato['bins_min'], bins_max=hotpotato['bins_max'],
                                    ducy_max=hotpotato['ducy_max'], deredden=hotpotato['do_deredden'], rmed_width=hotpotato['rmed_width'],
                                    already_normalised=False)
            periodograms.append(pgram)
            max_snrs.append(pgram.snrs.max())
            detrended_ts.append(dts)

        # Maximum S/N to be shown on periodogram plot
        snr_max = 1.25*np.max(max_snrs)
        # Produce candidate plot and save plot to disk.
        plot_name = hotpotato['PLOT_DIR'] + '/' + hotpotato['basename'] + '_ch%d'% (chan) + '_code%s'% (code) +'_period%.4f'% (period)
        candverf_plot(period, bins, detrended_ts, periodograms, hotpotato['beam_labels'],
                      start_mjds, snr_max, hotpotato['periodaxis_log'], plot_name, hotpotato['plot_formats'])

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
    # Default annotation labels
    if hotpotato['beam_labels']=='' or hotpotato['beam_labels']==[]:
        hotpotato['beam_labels'] = ['']*len(hotpotato['datafile_list'])
    # Default list of output plot formats = ['.png']
    if hotpotato['plot_formats']=='':
        hotpotato['plot_formats'] = ['.png']
    # Default output directory = DATA_DIR
    if hotpotato['PLOT_DIR']=='':
        hotpotato['PLOT_DIR'] = hotpotato['DATA_DIR']
    # Default log scale for period axis in periodogram = True
    if hotpotato['periodaxis_log']=='':
        hotpotato['periodaxis_log'] = True
    # Default minimum period covered in FFA search = 10 s
    if hotpotato['min_period']=='':
        hotpotato['min_period'] = 10.0
    # Default maximum period covered in FFA search = 100 s
    if hotpotato['max_period']=='':
        hotpotato['max_period'] = 100.0
    # Default fpmin = 3
    if hotpotato['fpmin']=='':
        hotpotato['fpmin'] = 3
    # Default S/N threshold = 8.0
    if hotpotato['SNR_threshold']=='':
        hotpotato['SNR_threshold'] = 8.0
    # Default bins_min = 8
    if hotpotato['bins_min']=='':
        hotpotato['bins_min'] = 10
    # Default bins_max = 11
    if hotpotato['bins_max']=='':
        hotpotato['bins_max'] = 11
    # Default max duty cycle = 0.5
    if hotpotato['ducy_max']=='':
        hotpotato['ducy_max'] = 0.5
    # Detrending flag
    if hotpotato['do_deredden']=='':
        hotpotato['do_deredden'] = False
    # Default running median window width = 12 s
    if hotpotato['rmed_width']=='':
        hotpotato['rmed_width'] = 12.0
    # Default memory load size = 1 GB
    if hotpotato['mem_load']=='':
        hotpotato['mem_load'] = 1.0
    return hotpotato
##############################################################
def main():
    """ Command line tool for running plot_cands.py """
    parser = ArgumentParser(description="Produce candidate verification plots.")
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
