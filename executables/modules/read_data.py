# Routines to read data from a blimpy data product.

from blimpy import Waterfall
##########################################################################
def read_watfile(datafile, mem_load):
    """
    Read in a .h5 or .fil file as a blimpy Waterfall object.

    Parameters
    ----------
    datafile : string
         Name of data file to load

    mem_load: float
         Maximum data size in GB allowed in memory (default: 1 GB)

    Returns
    -------
    wat : class object
        Blimpy Waterfall object of data file contents
    """
    wat = Waterfall(datafile, max_load=mem_load)
    return wat
##########################################################################
