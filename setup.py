"""
setup.py -- setup script for installation and use of packages
"""
import os
from setuptools import setup, find_packages, Command
from setuptools.command import install

__version__ = '0.1.0'

class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        os.system('rm -vrf ./build ./dist ./*.pyc ./*.tgz ./*.egg-info')

install_requires = [
        'astropy>=4.0',
        'blimpy>=2.0.0',
        'matplotlib>=3.1.0',
        'mpi4py>=3.1.1',
        'numpy>=1.18.1',
        'riptide-ffa>=0.2.3',
        'scipy>=1.6.0',
        'tqdm>=4.32.1'
]

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='blipss',
      version=__version__,
      author='Akshay Suresh',
      author_email='as3655@cornell.edu',
      description='Breakthrough Listen Investigation for Periodic Spectral Signals',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/akshaysuresh1/blipss',
      install_requires=install_requires,
      packages=find_packages(),
      license='MIT License',
      classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "License :: OSI Approved :: BSD 3-Clause License",
        "Topic :: Scientific/Engineering :: Astronomy"
      ],
      cmdclass={'clean': CleanCommand}
)
