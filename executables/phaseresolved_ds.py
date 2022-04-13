#!/usr/bin/env python
'''
# Produce a plot of a phase-resolved spectrum for a given folding period.

Run using the following syntax.
python phaseresolved_ds.py -i <Configuration script of inputs> | tee <Log file>
'''
from __future__ import print_function
from __future__ import absolute_import
# Custom imports
from modules.general_utils import setup_logger_stdout, create_dir
from modules.read_config import read_config
from modules.read_data import read_watfile
from modules.plotting import plot_phaseds
# Standard packages
from argparse import ArgumentParser
from riptide import TimeSeries
from tqdm import tqdm
import os, logging, time, sys
import numpy as np
##############################################################
def myexecute(inputs_cfg):
    """
    Primary function that handles script execution.

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

    # Read data.
    logger.info('Reading file: %s'% (hotpotato['datafile']))
    wat = read_watfile(hotpotato['DATA_DIR']+'/'+hotpotato['datafile'], hotpotato['mem_load'])
    # Extract relevant metadata from header.
    start_MJD = wat.header['tstart'] # Start MJD (UTC)
    tsamp = wat.header['tsamp'] # Sampling time (s)
    freqs_MHz = wat.header['fch1'] + np.arange(wat.header['nchans'])*wat.header['foff'] # 1D array of radio frequencies (MHz)

    # Default data shape in Waterfall objects = (nsamples, npol, nchans)
    # Reshape data to (nchans, nsamples), assuming index 0 of npol refers to Stokes-I.
    data = wat.data[:,0,:].T
    # Invert band if channel bandwidth is negative.
    if wat.header['foff'] < 0:
        data = np.flip(data, axis=0)
        freqs_MHz = np.flip(freqs_MHz)

    # Clip off edge channels.
    if hotpotato['stop_ch'] is None:
        hotpotato['stop_ch'] = len(data)
    # Start channel included, stop channel excluded.
    data = data[ hotpotato['start_ch'] : hotpotato['stop_ch'] ]
    freqs_MHz = freqs_MHz[ hotpotato['start_ch'] : hotpotato['stop_ch'] ]

    phaseresolved_ds =  np.zeros((len(data), hotpotato['bins']))
    logger.info('Computing phase-resolved spectrum')
    # Loop over channels.
    for j in tqdm(range(len(data))):
        ts = TimeSeries.from_numpy_array(data[j], tsamp=tsamp)
        # Detrend time series.
        if hotpotato['do_deredden']:
            ts = ts.deredden(hotpotato['rmed_width'])
        # Normalize time series to zero median and unit standard deviation.
        ts = ts.normalise()
        # Fold time series at user-specified period.
        phaseresolved_ds[j]= ts.fold(hotpotato['period'], hotpotato['bins'], subints=1)

    # Plot phase-resolved dynamic spectrum.
    create_dir(hotpotato['PLOT_DIR'])
    logger.info('Saving plot to disk')
    plot_name = hotpotato['PLOT_DIR'] + '/' +  hotpotato['basename'] + '_period%.5f'% (hotpotato['period'])
    plot_phaseds(phaseresolved_ds, freqs_MHz, hotpotato['period'], start_MJD, plot_name, hotpotato['plot_formats'])

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
    # Default plot format = ['.png']
    if hotpotato['plot_formats']=='' or hotpotato['plot_formats']==[]:
        hotpotato['plot_formats'] = ['.png']
    # Default output path for plots
    if hotpotato['PLOT_DIR']=='':
        hotpotato['PLOT_DIR'] = hotpotato['DATA_DIR']
    # Start channel for FFA search
    if hotpotato['start_ch']=='':
        hotpotato['start_ch'] = 0
    # Stop channel for FFA search
    if hotpotato['stop_ch']=='':
        hotpotato['stop_ch'] = None
    # Default folding period = 1 s
    if hotpotato['period']=='':
        hotpotato['period'] = 1.0
    # By default, use 10 bins across the folded profile.
    if hotpotato['bins']=='':
        hotpotato['bins'] = 10
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
    """ Command line tool for running phaseresolved_ds.py """
    parser = ArgumentParser(description="Produce a plot of a phase-resolved spectrum for a given folding period.")
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
