#!/usr/bin/env python
"""
Simulate an artificial data set and save it to disk as a filterbank file.

Run using the following syntax.
python simulate_data.py -i <Configuration script of inputs> | tee <Log file>
"""
from __future__ import print_function
from __future__ import absolute_import
# Custom packages
from modules.read_config import read_config
from modules.general_utils import setup_logger_stdout, create_dir
# Standard imports
from blimpy import Waterfall
from blimpy.io.sigproc import generate_sigproc_header
from riptide import TimeSeries
from argparse import ArgumentParser
import os, logging, time, sys
import numpy as np
##############################################################
def myexecute(inputs_cfg):
    """
    Primary function that handles synthetic data generation.

    Parameters
    ----------
    inputs_cfg : str
         Path to configuration script of inputs
    """
    # Profile code execution.
    prog_start_time = time.time()

    # Read inputs from config file and set default parameter values, if applicable.
    hotpotato = read_config(inputs_cfg)
    hotpotato = set_defaults(hotpotato)
    logger = setup_logger_stdout() # Set logger output to stdout().

    # No. of channels in which to inject a periodic signal
    N_inject = len(hotpotato['inject_channels'])

    # Generate Gaussian white noise background with zero mean and unit variance.
    data = np.random.randn(hotpotato['N_chans'], hotpotato['N_samples'])
    logger.info('Background Gaussian white noise data generated.')
    # Inject periodic signals.
    for i in range(N_inject):
        chan = hotpotato['inject_channels'][i] # Channel into which a periodic signal must be injected
        data[chan]  = TimeSeries.generate(length=hotpotato['N_samples']*hotpotato['t_samp'], tsamp=hotpotato['t_samp'], period=hotpotato['periods'][i],
                                  phi0=hotpotato['initial_phase'][i], ducy=hotpotato['duty_cycles'][i],
                                  amplitude=hotpotato['pulse_amplitudes'][i], stdnoise=1.0).data
        logger.info('Injected P = %.2f s signal into channel %d.'% (hotpotato['periods'][i], chan))
    data = data.T.reshape((data.T.shape[0], 1, data.T.shape[1])) # New data shape = (Nsamples, Npol, Nchans)

    # Build file header.
    header = {'machine_id': 0, 'telescope_id': 0, 'data_type': 1, 'nbits': 32, 'nifs': 1}
    header['fch1'] = hotpotato['fch1']
    header['foff'] = hotpotato['foff']
    header['tsamp'] = hotpotato['t_samp']
    header['nchans'] = hotpotato['N_chans']
    header['nsamples'] = hotpotato['N_samples']
    header['source_name'] = hotpotato['source_name']
    header['tstart'] = hotpotato['tstart']

    create_dir(hotpotato['OUTPUT_DIR'])
    # Construct a Waterfall object that will be written to disk as a filterbank file.
    wat = Waterfall() # Empty Waterfall object
    wat.header = header
    filename_out = hotpotato['OUTPUT_DIR']+ '/' + hotpotato['basename'] + '.fil'
    with open(filename_out, 'wb') as fh:
        logger.info('Writing filterbank file to %s'% (hotpotato['OUTPUT_DIR']))
        fh.write(generate_sigproc_header(wat)) # Trick Blimpy into writing a sigproc header.
        np.float32(data.ravel()).tofile(fh)

    # Calculate total run time for the code.
    prog_end_time = time.time()
    run_time = (prog_end_time - prog_start_time)/60.0
    logger.info('Code run time = %.5f minutes'% (run_time))

def set_defaults(hotpotato):
    """
    Set default values for keys in a dictionary of input parameters.

    Parameters
    ----------
    hotpotato : dictionary
         Dictionary of input parameters gathered from a configuration script
    """
    if hotpotato['inject_channels']=='':
        hotpotato['inject_channels'] = []
    if hotpotato['periods']=='':
        hotpotato['periods'] = []
    if hotpotato['duty_cycles']=='':
        hotpotato['duty_cycles'] = []
    if hotpotato['pulse_amplitudes']=='':
        hotpotato['pulse_amplitudes'] = []
    if hotpotato['initial_phase']=='':
        hotpotato['initial_phase'] = []
    if hotpotato['source_name']=='':
        hotpotato['source_name'] = 'Unknown'
    if hotpotato['tstart']=='':
        hotpotato['tstart'] = 0.0
    return hotpotato

##############################################################
def main():
    """ Command line tool for running rfifind. """
    parser = ArgumentParser(description="Generate an artificial data set.")
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
