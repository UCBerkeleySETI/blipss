# Routines to read data from a blimpy data product.

from blimpy import Waterfall
import numpy as np
##########################################################################
# Function to read an input file using blimpy routines.
def read_blimpy_file(datafile):
    obs = Waterfall(datafile)
    data = np.squeeze(obs.data)
    header = obs.header
    return data, header

##########################################################################
