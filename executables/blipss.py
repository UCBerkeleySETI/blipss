#!/usr/bin/env python
'''
Run channel-wise FFA on a set of input files, and output one .csv file of candidates per input file.

Run using the following syntax.
mpiexec -n (nproc) python -m mpi4py blipss.py -i <Configuration script of inputs> | tee <Log file>
'''
from __future__ import print_function
from __future__ import absolute_import
# Import custom modules.
from modules.general_utils import create_dir, setup_logger_stdout
from modules.helper_func import periodic_helper
from modules.read_config import read_config
from modules.read_data import read_watfile
# Load standard pacakages.
from blimpy import Waterfall
from mpi4py import MPI
import numpy as np
import os, logging, time, sys, glob, csv
from argparse import ArgumentParser
#########################################################################
def myexecute(datafile, hotpotato, logger):
    """
    Primary executable function called on each core. Reads input data, executes channel-wise FFA, removes harmonics, and save candidate information to disk.

    Parameters
    ----------
    datafile : string
         Name (including path) of input data file. Data file format must be readable by blimpy.

    hotpotato : dictionary
         Dictionary of input parameters read from a configuration script
    """
    # Extract basename from input data file name.
    supported_ext = ['.fil', '.h5'] # File extensions for filterbank and hd5 files
    if '/' in datafile:
        basename = datafile.split('/')[-1]
    for ext in supported_ext:
        if ext in basename:
            basename = basename.split(ext)[0]

    # Read input data as a Waterfall object.
    # NOTE: In future, the following line can be replaced by a more efficient read functionality, if available.
    wat = read_watfile(datafile, hotpotato['mem_load'])

    # Default data shape in Waterfall objects = (nsamples, npol, nchans)
    # Reshape data to (nchans, nsamples), assuming index 0 of npol refers to Stokes-I.
    data = wat.data[:,0,:].T

    # Clip off edge channels.
    if hotpotato['stop_ch'] is None:
        hotpotato['stop_ch'] = len(data)
    # Start channel included, stop channel excluded.
    data = data[ hotpotato['start_ch'] : hotpotato['stop_ch'] ]

    # Extract relevant metadata from header.
    tsamp = wat.header['tsamp'] # Sampling time (s)
    freqs_MHz = wat.header['fch1'] + np.arange()*wat.header['foff'] # 1D array of radio frequencies (MHz)

    # Run channel-wise FFA and return properties of detected candidates.
    logger.info('Running channel-wise FFA on basename %s'% (basename))
    cand_chans, cand_periods, cand_snrs, cand_bins, cand_best_widths = periodic_helper(data, hotpotato['start_ch'], tsamp,
                                                                                       hotpotato['min_period'], hotpotato['max_period'], hotpotato['fpmin'],
                                                                                       hotpotato['bins_min'], hotpotato['bins_max'], hotpotato['ducy_max'],
                                                                                       hotpotato['do_deredden'], hotpotato['rmed_width'], hotpotato['SNR_threshold'],
                                                                                       hotpotato['mem_load'])
    logger.info('%d canidates found in %s'% (basename))
    logger.info('FFA completed on %s'% (basename))

    # Convert channel numbers to radio frequencies (MHz).
    cand_radiofreqs = freqs_MHz[cand_chans]

    # TO DO: IDENTIFY AND LABEL HARMONICS ('H' for harmonic, 'F' for fundamental)
    cand_harmonicflag = np.array(['H']*len(cand_chans))

    # Scatter plot of candidate S/N in radio frequency vs. trial period diagram.
    if hotpotato['do_plot']:
        logger.info('Producing scatter plot for %s'% (basename))

    # Construct output .csv file name.
    output_csv_name = hotpotato['OUTPUT_DIR'] + '/' + basename + '_cands.csv'
    # Define structure of .csv file.
    header = ['Channel', 'Radio frequency (MHz)', 'Bins', 'Best width', 'Period (s)', 'S/N', 'Harmonic?']
    columns = [cand_chans, cand_radiofreqs, cand_bins, cand_best_widths, cand_periods, cand_snrs, cand_harmonicflag]
    zipped_rows = zip(*columns)

    # Write .csv file with specified columns.
    logger.info('Writing .csv output: %s'% (output_csv_name))
    with open(output_csv_name,'w') as csvfile:
    	writer = csv.writer(csvfile,delimiter=',')
    	writer.writerow(header)
    	for line in zipped_rows:
    			writer.writerow(line)

#########################################################################
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
    # Default output path
    if hotpotato['OUTPUT_DIR']=='':
        hotpotato['OUTPUT_DIR'] = hotpotato['DATA_DIR']
    # Default plotting setting = False
    if hotpotato['do_plot']=='':
        hotpotato['do_plot'] = False
    # Default plot format = ['.png']
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
        file_list = sorted(glob.glob(hotpotato['DATA_DIR'] + '/' + hotpotato['glob_input']))
        N_files = len(file_list)
        parent_logger.info('Total no. of input files = %d'% (N_files))

        if nproc>1:
            # In case of multiple processors, the parent processor distributes calls evenly between the child processors and itself.
            distributed_file_list = np.array_split(np.array(file_list), nproc)
            # Send calls to child processors.
            for indx in range(1,nproc):
                comm.send( (distributed_file_list[indx-1], hotpotato), dest=indx, tag=indx)
            # Run tasks assigned to parent processor.
            for datafile in distributed_file_list[-1]:
                myexecute(datafile, hotpotato, parent_logger)
            comm.Barrier() # Wait until all processors to complete their respective calls.
        else:
            # In case nproc =1 , the available processor run FFA on input data files in serial fashion.
            for datafile in file_list:
                myexecute(datafile, hotpotato, parent_logger)

        # Calculate total run time for the code.
        prog_end_time = time.time()
        run_time = (prog_end_time - prog_start_time)/60.0
        parent_logger.info('Code run time = %.3f minutes'% (run_time))
        parent_logger.info('FINISHING RANK: 0')
    else:
        child_logger = setup_logger_stdout() # Set up separate logger for each child processor.
        child_logger.info('STARTING RANK: %d'% (rank))
        # Recieve data from parent processor.
        child_file_list, hotpotato = comm.recv(source=0, tag=rank)
        # Run process.
        for datafile_child in child_file_list:
            myexecute(datafile_child, hotpotato, child_logger)
        # Intimate task completion on child processor.
        child_logger.info('FINISHING RANK: %d'% (rank))
        comm.Barrier() # Wait for all processors to complete their respective calls.
#########################################################################
def usage():
    return """
usage: mpiexec -n (nproc) python -m mpi4py blipss.py [-h] -i INPUTS_CFG

Run the fast folding algorithm (FFA) on a per-channel basis for a set of input files, and output one .csv file of candidates per input file.

required arguments:
-i INPUTS_CFG  Configuration script of inputs
optional arguments:
-h, --help     show this help message and exit
    """
##############################################################################
def main():
    """ Command line tool for BLIPSS"""
    parser = ArgumentParser(description="Breakthrough Listen Investigation for Periodic Spectral Signals (BLIPSS)",usage=usage(),add_help=False)
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
