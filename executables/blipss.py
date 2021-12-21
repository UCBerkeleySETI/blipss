#!/usr/bin/env python
'''
Primary executable script for the Breakthrough Listen Investigation for Periodic Spectral Signals (BLIPSS)

'''
from __future__ import print_function
from __future__ import absolute_import
# Import custom modules.
from modules.read_config import read_config
from modules.general_utils import create_dir, setup_logger_stdout
from modules.helper_func import periodic_helper
# Load standard pacakages.
from blimpy import Waterfall
from mpi4py import MPI
import numpy as np
import os, logging, time, sys, glob
#########################################################################
# Set up default values for keys of the dictionary "hotpotato".
def set_defaults(hotpotato):
    if hotpotato['OUTPUT_DIR']=='':
        hotpotato['OUTPUT_DIR'] = hotpotato['DATA_DIR']
    # Start channel for FFA search
    if hotpotato['start_ch']=='':
        hotpotato['start_ch'] = 0
    # Stop channel for FFA search:
    if hotpotato['stop_ch']=='':
        hotpotato['stop_ch'] = 0        
    # Default minimum period covered in FFA search = 0.1 s
    if hotpotato['min_period']=='':
        hotpotato['min_period'] = 0.1
    # Default maximum period covered in FFA search = 10 s
    if hotpotato['max_period']=='':
        hotpotato['max_period'] = 10.0
    # Default S/N threshold = 10.0
    if hotpotato['SNR_threshold']=='':
        hotpotato['SNR_threshold'] = 10.0
    # Default bins_min = 100
    if hotpotato['bins_min']=='':
        hotpotato['bins_min'] = 100
    # Default bins_max = 110
    if hotpotato['bins_max']=='':
        hotpotato['bins_max'] = 110
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
        print('STARTING RANK 0')
        # Profile code execution.
        prog_start_time = time.time()

        parse_args = parser.parse_args()
        # Initialize parameter values
        inputs_cfg = parse_args.inputs_cfg

        # Read inputs from config file and set default parameter values, if applicable.
        hotpotato = read_config(inputs_cfg)
        hotpotato = set_defaults(hotpotato)
        parent_logger = setup_logger_stdout() # Set logger output to stdout().

        # Create output path if non-existent.
        create_dir(hotpotato['OUTPUT_DIR'])

        # List of ON files
        ON_files = sorted(glob.glob(hotpotato['DATA_DIR'] + '/' + hotpotato['on_files_glob']))
        N_on = len(ON_files)
        print('No. of ON files = %d'% (N_on))
        # List of OFF files
        OFF_files = sorted(glob.glob(hotpotato['DATA_DIR'] + '/' + hotpotato['off_files_glob']))
        N_off = len(OFF_files)
        print('No. of OFF files = %d'% (N_off))

        # Calculate total run time for the code.
        prog_end_time = time.time()
        run_time = (prog_end_time - prog_start_time)/60.0
        parent_logger.info('Code run time = %.5f minutes'% (run_time))
        print('FINISHING RANK 0')
    else:
        # Receive data from parent processsor.


#########################################################################
def usage():
    return """
usage: nice -(nice value) mpiexec -n (nproc) python -m mpi4py blipss.py [-h] -i INPUTS_CFG
Argmunents in parenthesis are required numbers for an MPI run.

BLIPSS applies a fast folding algorithm on a per-channel basis to detect periodic signals in dynamic spectra .

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

'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import pandas as pd

import argparse
import time
from tqdm import tqdm
import multiprocessing as mp

from blimpy import Waterfall
from riptide import TimeSeries, ffa_search

start = time.time()
parser = argparse.ArgumentParser()
parser.add_argument('signal', type = str, help = 'Location of ON data file.')
parser.add_argument('--background', action = "append", type = str, default = None, help = 'Location of OFF data file.')
parser.add_argument('--cutoff', type = float, default = 10, help = 'SNR cutoff value.')
parser.add_argument('--alias', type = int, default = 1, help = 'Number of periods to check for harmonics.')
parser.add_argument('--range', type = float, nargs = 2, default = [0.1, 10], help = 'Period range for FFA search.')
parser.add_argument('--multi', action = "store_true", help = 'Use multiprocessing.')
parser.add_argument('--simulate', action = "store_true", help = 'Turns on simulation of fake signal.')
parser.add_argument('--beam', action = "store_true", help = 'Creates a three-digit code summarizing ON-OFF comparison.')
parser.add_argument('--output', type = str, default = "signal.txt", help = 'Name of output file.')
args = parser.parse_args()
on_file = args.signal
off_files = args.background
cutoff = args.cutoff
num_periods = args.alias
period_range = args.range
multi = args.multi
simulate = args.simulate
beam = args.beam
output = args.output


def periodic_analysis(on_data, off_data, freqs, nchans, tsamp, start, stop, cutoff):
    """Periodic analysis on a set of ON-OFF files."""
    periods = []
    frequencies = []
    snrs = []
    best_periods = []
    indicators = []
    best_widths = []
    min_widths = []
    max_widths = []
    all_codes = []

    for i in range(int(start * nchans), int(stop * nchans)):

        if sum(on_data[:, i]) == 0:
            continue
        on_periods, on_freqs, sn_ratios, best_period, widths, mini, maxi = periodic_helper(on_data[:, i], freqs[i], tsamp, cutoff)
        if off_data is not None:
            indicator = np.zeros(len(on_periods))
            codes = np.zeros(len(on_periods), dtype = str)
            for j in range(len(off_data)):
                datum = off_data[j]
                off_periods = periodic_helper(datum[:, i], freqs[i], tsamp, cutoff, False)
                if beam:
                    indicator, codes = compare_on_off(on_periods, off_periods, indicator, codes)
                    prev = '' + '0' * j
                    if prev in codes:
                        codes = [s + '0' for s in codes]
                else:
                    indicator = compare_on_off(on_periods, off_periods, indicator)

        periods.extend(on_periods)
        frequencies.extend(on_freqs)
        snrs.extend(sn_ratios)
        best_periods.append(best_period)
        best_widths.extend(widths)
        min_widths.extend(mini)
        max_widths.extend(maxi)
        if off_data is not None:
            indicators.extend(indicator)
            if beam:
                all_codes.extend(codes)

    if off_data is None:
        return periods, frequencies, snrs, best_periods, best_widths, min_widths, max_widths
    else:
        if not beam:
            return periods, frequencies, snrs, best_periods, best_widths, min_widths, max_widths, indicators
        return periods, frequencies, snrs, best_periods, best_widths, min_widths, max_widths, indicators, all_codes

def compare_on_off(on_periods, off_periods, indicator, codes = None):
    """"Compares ON and OFF files."""
    counter, tol = 0, 1e-4
    for i in range(len(on_periods)):
        po = on_periods[i]
        for j in range(len(off_periods)):
            pf = off_periods[j]
            if abs(po - pf) <= tol:
                if indicator[counter] == 0:
                    indicator[counter] = 1
                if codes is not None:
                    codes[counter] += '1'
            else:
                if codes is not None:
                    codes[counter] += '0'
        counter += 1
    if codes is not None:
        return indicator, codes
    return indicator


def find_harmonics(periods, best_periods, num_periods):
    """Finds and labels harmonics in periodograms."""
    harmonics = np.zeros(len(periods), dtype = bool)

    ranked = pd.Series(np.round(np.array(best_periods), 4)).value_counts()
    inspect = ranked.keys()[0:num_periods]
    counts = np.zeros(len(inspect))
    for i in range(len(inspect)):
        for j in range(len(periods)):
            check = (round(periods[j] / inspect[i]) > 1)
            close = (abs((periods[j] / inspect[i]) - round(periods[i] / inspect[i])) <= 1e-3)
            if check and close:
                counts[i] += 1
                if not harmonics[j]:
                    harmonics[j] = True

    return ranked, counts, harmonics


def concat_helper(results):
    """Concatenates results from worker processes."""
    periods = []
    frequencies = []
    snrs = []
    best_periods = []
    widths = []
    min_widths = []
    max_widths = []
    indicators = []
    codes = []

    for package in results:
        periods.extend(package[0])
        frequencies.extend(package[1])
        snrs.extend(package[2])
        best_periods.extend(package[3])
        widths.extend(package[4])
        min_widths.extend(package[5])
        max_widths.extend(package[6])
        if off_files is not None:
            indicators.extend(package[7])
            if beam:
                codes.extend(package[8])

    final = [np.array(periods), np.array(frequencies), np.array(snrs), np.array(best_periods)]
    final.extend([np.array(widths), np.array(min_widths), np.array(max_widths)])
    if off_files is not None:
        final.append(np.array(indicators))
        if beam:
            final.append(np.array(codes))
    return final


def plot_helper(periods, frequencies, snrs, harmonics, indicators):
    """Plots frequency channel vs. periodogram."""

    full_signal = list(zip(periods, frequencies, snrs))
    filter = np.zeros(len(full_signal), dtype = bool)
    signal = []
    alias = []
    background = []
    for i in range(len(harmonics)):
        if harmonics[i]:
            alias.append(full_signal[i])
        else:
            if indicators[i]:
                background.append(full_signal[i])
            else:
                filter[i] = 1
                signal.append(full_signal[i])

    cmap = plt.cm.viridis
    norm = matplotlib.colors.Normalize(vmin = min(snrs), vmax = max(snrs))

    plt.figure(figsize = (8, 6))
    if len(signal) > 0:
        plt.scatter(list(zip(*signal))[0], list(zip(*signal))[1], c = cmap(norm(list(zip(*signal))[2])), marker = 'o')
    if len(alias) > 0:
        plt.scatter(list(zip(*alias))[0], list(zip(*alias))[1], c = cmap(norm(list(zip(*alias))[2])), marker = '+')
    if len(background) > 0:
        plt.scatter(list(zip(*background))[0], list(zip(*background))[1], c = cmap(norm(list(zip(*background))[2])), marker = '^')

    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap = cmap, norm = norm))
    plt.xlabel('Periods')
    plt.ylabel('Frequencies')
    cbar.set_label('SNR', rotation = 270)
    plt.savefig('output.png')
    plt.show();

    return filter, signal


obs = Waterfall(on_file)
data = np.squeeze(obs.data)

freqs = np.array([obs.header['fch1'] + i * obs.header['foff'] for i in range(obs.header['nchans'])])
nchans, tsamp = obs.header['nchans'], obs.header['tsamp']

if simulate:
    injection = np.random.choice(freqs, 1, replace = False)
print("Progress: Read ON file.")

background_data = None
if off_files is not None:
    background_data = []
    for off_file in off_files:
        background = Waterfall(off_file)
        back_data = np.squeeze(background.data)
        background_data.append(back_data)
    print("Progress: Read OFF files.")

if multi:
    pool = mp.Pool(mp.cpu_count())
    on_iterables = [(data, background_data, freqs, nchans, tsamp, 0.1 * i, 0.1 * (i + 1), cutoff) for i in range(1, 9)]
    on_results = pool.starmap(periodic_analysis, on_iterables)
    on_results = concat_helper(on_results)
else:
    on_results = periodic_analysis(data, background_data, freqs, nchans, tsamp, 0.1, 0.9, cutoff)
    on_results = concat_helper([on_results])

ranked, counts, harmonics = find_harmonics(on_results[0], on_results[3], num_periods)
print("Progress: File processing complete.")

if off_files is not None:
    filter, signal = plot_helper(on_results[0], on_results[1], on_results[2], harmonics, on_results[-2])
else:
    filter, signal = plot_helper(on_results[0], on_results[1], on_results[2], harmonics, np.zeros(len(harmonics)))

for i in tqdm(range(len(signal))):
    aliasing, round_period = 0, round(signal[i][0], 4)
    if round_period in ranked.keys()[0:num_periods]:
        aliasing = counts[np.where(ranked.keys() == round_period)]
    signal[i] += (np.where(freqs == signal[i][1])[0][0], abs(obs.header['foff']), aliasing, period_range[0], period_range[1])
    signal[i] += (on_results[4][filter][i], on_results[5][filter][i], on_results[6][filter][i])
    signal[i] += (obs.header['source_name'], obs.header['tstart'], on_file)
    if beam:
        signal[i] += (on_results[-1][i],)
np.savetxt(output, signal, fmt = "%s")

if multi:
    pool.close()
    pool.join()

end = time.time()
print('Best Period: ', ranked.keys()[0])
print('Time Taken: ', end - start)
'''