# General file handling utilities

import os, sys, logging
###################################################
# Create a directory if non-existent.
def create_dir(folder):
    if not os.path.isdir(folder):
        print('%s does not exist.'% (folder))
        print('Creating %s'% (folder))
        os.makedirs(folder)
###################################################
# Set up logger output to stdout.
def setup_logger_stdout(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s'):
    root = logging.getLogger('main')
    root.propagate = False
    root.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %I:%M:%S %p')
    handler.setFormatter(formatter)
    root.addHandler(handler)

    return root
###################################################
