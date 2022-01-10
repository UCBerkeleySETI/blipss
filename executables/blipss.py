#!/usr/bin/env python
'''
Primary executable script for the Breakthrough Listen Investigation for Periodic Spectral Signals (BLIPSS)

Run using the following syntax.
mpiexec -n (nproc) python -m mpi4py blipss.py -i <Configuration script of inputs> | tee <Log file>
'''
from __future__ import print_function
from __future__ import absolute_import
# Import custom modules.
from modules.general_utils import create_dir, setup_logger_stdout
from modules.helper_func import periodic_helper
from modules.read_config import read_config
# Load standard pacakages.
from blimpy import Waterfall
from mpi4py import MPI
import numpy as np
import os, logging, time, sys, glob
from argparse import ArgumentParser
#########################################################################
# Set up default values for keys of the dictionary "hotpotato".
def set_defaults(hotpotato):
    """
    Set default values for keys in a dictionary of input parameters.

    Parameters
    ----------
    hotpotato : dictionary
         Dictionary of input parameters read from a configuration script

    Returns
    -------
    hotpotato : dictionary
        Input dictionary with keys set to default values
    """
    # Availability of OFF data
    if hotpotato['have_off']=='':
        hotpotato['have_off'] = False
    # Default output path
    if hotpotato['OUTPUT_DIR']=='':
        hotpotato['OUTPUT_DIR'] = hotpotato['DATA_DIR']
    # Default plot format = '.png'
    if hotpotato['plot_formats']=='' or hotpotato['plot_formats']==[]:
        hotpotato['plot_formats'] = ['.png']
    # Start channel for FFA search
    if hotpotato['start_ch']=='':
        hotpotato['start_ch'] = 0
    # Stop channel for FFA search:
    if hotpotato['stop_ch']=='':
        hotpotato['stop_ch'] = None
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
#########################################################################
# Run BLIPSS on data.
def __MPI_MAIN__(parser):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    stat = MPI.Status()
    nproc = comm.Get_size()

    # Parent processor
    if rank==0:
        parent_logger = setup_logger_stdout() # Set logger output to stdout().
        parent_logger.info('STARTING RANK: 0')
        # Profile code execution.
        prog_start_time = time.time()

        parse_args = parser.parse_args()
        # Initialize parameter values
        inputs_cfg = parse_args.inputs_cfg

        # Read inputs from config file and set default parameter values, if applicable.
        hotpotato = read_config(inputs_cfg)
        hotpotato = set_defaults(hotpotato)

        # Create output path if non-existent.
        create_dir(hotpotato['OUTPUT_DIR'])

        # List of ON files
        ON_files = sorted(glob.glob(hotpotato['DATA_DIR'] + '/' + hotpotato['on_files_glob']))
        N_on = len(ON_files)
        parent_logger.info('No. of ON files = %d'% (N_on))
        # List of OFF files
        if hotpotato['have_off']:
            OFF_files = sorted(glob.glob(hotpotato['DATA_DIR'] + '/' + hotpotato['off_files_glob']))
            N_off = len(OFF_files)
            parent_logger.info('No. of OFF files = %d'% (N_off))
        else:
            parent_logger.info('Zero OFF files supplied.')
            OFF_files = []
            N_off = 0
        # Gather all data file info.
        datafiles_list = ON_files + OFF_files
        file_labels = ['ON']*N_on + ['OFF']*N_off
        N_files = N_on + N_off

        if nproc==1:
            # In case nproc =1 , the available processor run FFA on input data files in serial fashion.
            for indx in range(N_files):
                select_chans, select_radiofreqs, periods, snrs, best_widths, min_radiofreq, max_radiofreq = periodic_helper(datafiles_list[indx], hotpotato['start_ch'], hotpotato['stop_ch'],
                                                                                                                            hotpotato['min_period'], hotpotato['max_period'], hotpotato['fpmin'],
                                                                                                                            hotpotato['bins_min'], hotpotato['bins_max'], hotpotato['ducy_max'],
                                                                                                                            hotpotato['do_deredden'], hotpotato['rmed_width'], hotpotato['SNR_threshold'],
                                                                                                                            hotpotato['mem_load'], return_radiofreq_limits=True)
                parent_logger.info('FFA search completed on %s file:\n %s'% (file_labels[indx], datafiles_list[indx]))
        else:
            # In case of multiple processors, the parent processor distributes calls evenly between the child processors and itself.
            distributed_file_list = np.array_split(np.array(datafiles_list), nproc)
            distributed_label_list = np.array_split(np.array(file_labels), nproc)
            # Send calls to child processors.
            for indx in range(1,nproc):
                comm.send((distributed_file_list[indx-1], distributed_label_list[indx-1], hotpotato), dest=indx, tag=indx)
            # Run tasks assigned to parent processor.
            for j in range(len(distributed_file_list[-1])):
                select_chans, select_radiofreqs, periods, snrs, best_widths, min_radiofreq, max_radiofreq = periodic_helper(distributed_file_list[-1][j], hotpotato['start_ch'], hotpotato['stop_ch'],
                                                                                                                            hotpotato['min_period'], hotpotato['max_period'], hotpotato['fpmin'],
                                                                                                                            hotpotato['bins_min'], hotpotato['bins_max'], hotpotato['ducy_max'],
                                                                                                                            hotpotato['do_deredden'], hotpotato['rmed_width'], hotpotato['SNR_threshold'],
                                                                                                                            hotpotato['mem_load'], return_radiofreq_limits=True)
                parent_logger.info('FFA search completed on %s file:\n %s'% (distributed_label_list[-1][j], distributed_file_list[-1][j]))
            comm.Barrier() # Wait for all child processors to complete respective calls.

        # Calculate total run time for the code.
        prog_end_time = time.time()
        run_time = (prog_end_time - prog_start_time)/60.0
        parent_logger.info('Code run time = %.5f minutes'% (run_time))
        parent_logger.info('FINISHING RANK: 0')
    else:
        child_logger = setup_logger_stdout() # Set up separate logger for each child processor.
        # Receive data from parent processsor.
        child_logger.info('STARTING RANK: %d'% (rank))
        # Recieve data from parent processor.
        file_list, label_list, hotpotato = comm.recv(source=0, tag=rank)
        for counter in range(len(file_list)):
            select_chans, select_radiofreqs, periods, snrs, best_widths, min_radiofreq, max_radiofreq = periodic_helper(file_list[counter], hotpotato['start_ch'], hotpotato['stop_ch'],
                                                                                                                            hotpotato['min_period'], hotpotato['max_period'], hotpotato['fpmin'],
                                                                                                                            hotpotato['bins_min'], hotpotato['bins_max'], hotpotato['ducy_max'],
                                                                                                                            hotpotato['do_deredden'], hotpotato['rmed_width'], hotpotato['SNR_threshold'],
                                                                                                                            hotpotato['mem_load'], return_radiofreq_limits=True)
            child_logger.info('FFA search completed on %s file:\n %s'% (label_list[counter], file_list[counter]))
        child_logger.info('FINISHING RANK: %d'% (rank))
        comm.Barrier() # Wait for all processors to complete their respective calls.
#########################################################################
def usage():
    return """
usage: mpiexec -n (nproc) python -m mpi4py blipss.py [-h] -i INPUTS_CFG

BLIPSS runs a fast folding algorithm on a per-channel basis to detect periodic signals in dynamic spectra.

required arguments:
-i INPUTS_CFG  Configuration script of inputs
optional arguments:
-h, --help     show this help message and exit
    """
##############################################################################
def main():
    """ Command line tool for BLIPSS"""
    parser = ArgumentParser(description="Breakthrough Listen Investigation for Periodic Spectral Signals",usage=usage(),add_help=False)
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    required.add_argument('-i', action='store', required=True, dest='inputs_cfg', type=str,
                            help="Configuration script of inputs")
    parser._action_groups.append(optional)

    if len(sys.argv)==1:
        parser.print_help()
        sys.exit(1)

    # Run MPI-parallelized prepsubband.
    __MPI_MAIN__(parser)
##############################################################################
if __name__=='__main__':
    main()
##############################################################################
