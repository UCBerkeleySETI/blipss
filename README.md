# blipss
Breakthrough Listen Investigation for Periodic Spectral Signals (BLIPSS or `blipss`) is a software package that utilizes the Fast Folding Algorithm (FFA) in [`riptide`](https://github.com/v-morello/riptide) to search for channel-wide periodic signals in radio dynamic spectra.

---

## Table of Contents
- [Package dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)

## Package Dependencies <a name="dependencies"></a>
```blipss``` is written in Python 3, and has the following package dependencies.
- astropy >= 4.0
- [`blimpy`](https://github.com/UCBerkeleySETI/blimpy) >= 2.0.0
- matplotlib >= 3.1.0
- mpi4py >= 3.1.1
- numpy >= 1.18.1
- ['riptide'](https://github.com/v-morello/riptide) >= 0.2.3
- scipy >= 1.6.0
- tqdm >= 4.32.1

## Installation <a name="installation"></a>
1. Clone this repository to your local machine. To do so, execute the following at the command line.
```
git clone git@github.com:akshaysuresh1/blipss.git
```
2. Verify that your local Python 3 installation satisfies all dependencies of ```blipss```. If not, either manually install the missing dependencies or run the below calls.
```
cd blipss
python setup.py install
python setup.py clean
```

## Usage <a name="usage"></a>
TBD.

## Troubleshooting <a name="troubleshooting"></a>
Please submit an issue to voice any problems or requests.

Improvements to the code are always welcome.
