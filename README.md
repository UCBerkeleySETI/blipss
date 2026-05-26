# BLIPSS
[![AJ Paper](https://img.shields.io/badge/DOI-10.3847/1538--3881/acccf0-blue)](https://doi.org/10.3847/1538-3881/acccf0)
[![arXiv](http://img.shields.io/badge/astro.ph-2305.18527-B31B1B.svg)](https://arxiv.org/abs/2305.18527)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/UCBerkeleySETI/blipss/blob/main/LICENSE)

The Breakthrough Listen Investigation for Periodic Spectral Signals (BLIPSS) targets the detection of narrowband periodic radar transmissions from potential technologically advanced alien life forms residing in the Universe. See this [link](http://www.mobileradar.org/radar_descptn_3.html) for examples of historic terrestrial radar operating at different radio frequencies.

BLIPSS utilizes the Fast Folding Algorithm (FFA) in [`riptide-ffa`](https://github.com/v-morello/riptide) to search for channel-wide periodic signals in radio dynamic spectra.

## Citation

If using ``blipss`` contributes to a scientific publication, please cite the article:
[Suresh et al., <i>"A 4&ndash;8 GHz Galactic Center Search for Periodic Technosignatures"</i>. 2023 AJ 165 255](https://ui.adsabs.harvard.edu/abs/2023arXiv230518527S/abstract).

---

## Table of Contents
- [Installation](#installation)
    - [Option 1: Using a uv environment](#a-using-uv-recommended)
    - [Option 2: Using a conda environment](#b-using-a-conda-environment)
    - [Add-on: Enabling LaTeX in Plots](#enabling-latex-in-plots)
- [Repository Organization](#organization)
- [Functionalities and Usage](#usage)
    - [blipss.py](#blipss_exec)
    - [compare_cands.py](#comparecands)
    - [plot_cands.py](#plotcands)
    - [phaseresolved_ds.py](#phaseds)
    - [inject_signal.py](#injectsignal)
    - [simulate_data.py](#simulate-data)
- [Troubleshooting](#troubleshooting)

## Installation <a name="installation"></a>

Choose one of the two installation methods below.

### A. Using uv (Recommended)

The installation steps below assume that you have [uv](https://docs.astral.sh/uv/getting-started/installation/) and [git](https://git-scm.com/downloads) installed on your local machine.

1. Clone the repository to your local machine.
```bash
git clone git@github.com:UCBerkeleySETI/blipss.git
```

2. Navigate into the repository and install the package.
```bash
cd blipss
make install
```

3. Verify that the installation is complete by checking that all unit tests pass locally.
```bash
make test
```

4. Activate the created uv environment.
```bash
source .venv/bin/activate
```
Your uv environment is now ready for use.

### B. Using a conda environment

The installation steps below assume that you have [Anaconda](https://www.anaconda.com/download) and [git](https://git-scm.com/downloads) installed on your local machine.

1. Create a conda environment with Python 3.12.
```bash
conda create -n blipss-env python=3.12
```

2. Activate the conda environment.
```bash
conda activate blipss-env
```

3. Install `pybind11`, a prerequisite for [`riptide-ffa`](https://github.com/v-morello/riptide).
```bash
pip install pybind11
```

3. Clone the repository to your local mchine.
```bash
git clone git@github.com:UCBerkeleySETI/blipss.git
```

4. Navigate into the repository and install the full package. Y
```bash
cd blipss
pip install .
```
Your conda environment is now ready for use.

### Enabling LaTeX in Plots

BLIPSS plotting modules expose a boolean `use_latex` flag that, when set to `True`, renders axis labels and annotations using LaTeX. Enabling this flag requires that a LaTeX distribution be installed on your system.

Install a LaTeX distribution appropriate for your operating system.
- **macOS**: [MacTeX](https://www.tug.org/mactex/)
- **Linux**: [TeX Live](https://www.tug.org/texlive/) via your package manager (e.g., `sudo apt install texlive-full`)
- **Windows**: [MiKTeX](https://miktex.org/)

Also, ensure `dvipng` and `ghostscript` are available, as matplotlib requires them for LaTeX rendering.
```bash
# macOS (Homebrew)
brew install ghostscript

# Debian/Ubuntu
sudo apt install dvipng ghostscript
```

If you do not have a LaTeX distribution installed, leave `use_latex=False` (the defaul behavior). Matplotlib will use its built-in math renderer and plots will still be generated without error.

## Repository Organization <a name="organization"></a>

```
blipss/                            # repository root
├── blipss/                        # Python package
│   ├── cli/                       # CLI entry points (installed as console scripts)
│   ├── core/                      # FFA period-finding and harmonic detection algorithms
│   ├── io/                        # Data I/O: filterbank/HDF5 reading, YAML config parsing, filterbank writing
│   ├── models/                    # Pydantic data models
│   ├── plotting/                  # Plotting utilities
│   ├── utils/                     # General utilities
│   └── constants.py
├── config/                        # Sample YAML configuration files, one per CLI command
├── tests/                         # Test suite mirroring the blipss/ package layout
├── pyproject.toml
└── README.md
```

Each CLI command reads its parameters from a companion YAML file in `config/`. For example:
```bash
simulate-data --config config/simulate_data.yaml
```

## Functionalities and Usage <a name="usage"></a>
The BLIPSS package contains six executable scripts, which are:
1. ``blipss.py`` <a name="blipss_exec"></a> <br>
Executes channel-wise FFA on input data files (filterbank or hdf5), identifies harmonics of detected periods, and outputs a .csv file of candidates. Here is a schematic of the `blipss.py` workflow. <br>

![BLIPSS workflow (Jan 27, 2022)](https://github.com/UCBerkeleySETI/blipss/blob/main/images/blipss_design_2022Jan27.png?raw=True)

Columns in the .csv file output by ``blipss.py`` include 'Channel', 'Radio frequency (MHz)', 'Bins', 'Best width', 'Period (s)', 'S/N', and 'Harmonic flag'. <br>

The current implementation takes about 35 min. to run on a single mid-resolution filterbank product (1.07 s sampling, 2.86 kHz, 1703936 channels). For processing multiple input files in parallel, enable MPI via the following syntax.
```
mpiexec -n <nproc> python -m mpi4py executables/blipss.py -i config/blipss.cfg | tee <Log file>
```
The above syntax assumes a Python call from the repo base directory. Alter paths as required to supply executable and config scripts located in different directories.

---
2. ``compare_cands.py``: <a name="comparecands"></a>
Compare periodicity detections across a set of <em>N</em> .csv files generated by ``blipss.py``. For every unique candidate period, an <em>N</em>-digit binary code is generated, wherein ones and zeros represent detections and non-detections respectively.<br>

Note that the order of input .csv files passed to ``compare_cands.py`` matters. When read from left to right, the <em>i</em>-th place of the <em>N</em>-digit binary code refers to the <em>i</em>-th .csv file in the input list.<br>

The output from ``compare_cands.py`` is a single .csv file containing the following columns.<br>
'Channel', 'Radio frequency (MHz)', 'Bins', 'Best width', 'Period (s)', 'S/N', 'Code' <br>

Execution syntax from repo base folder:
```
python executables/compare_cands.py -i config/compare_cands.cfg | tee <Log file>
```

---
3. ``plot_cands.py``: <a name="plotcands"></a>
Produce verification plots for a chosen subset of candidates. <br>

Here's a sample plot of a candidate with period 30 s and code 101010. Each row represents a different data file. The left column shows periodograms derived from different data files. We indicate the candidate period by red dashed vertical lines in the left panels. The right column illustrates average pulse profiles and pulse stacks in the phase-time plane. <br>

![B04 candidate](https://github.com/UCBerkeleySETI/blipss/blob/main/images/sim_cand.png?raw=True)

Clearly, we see significant spikes at the expected candidate period in the periodograms on the first, third, and fifth rows. <br>

Execution syntax from repo base folder:
```
python executables/plot_cands.py -i config/plot_cands.cfg | tee <Log file>
```

---
4. ``phaseresolved_ds.py``: <a name="phaseds"></a>
Compute and plot the phase-resolved spectrum for a given folding period.

Here's a sample output showing a phase-resolved spectrum of pulsar B0355+54.

![psrB0355 spectrum](https://github.com/UCBerkeleySETI/blipss/blob/main/images/guppi_58702_22205_PSR_B0355%2B54_0041_period0.15637.png?raw=True)

Execution syntax from repo base folder:
```
python executables/phaseresolved_ds.py -i config/phaseresolved_ds.cfg | tee <Log file>
```

---
5. ``inject_signal.py``: <a name="injectsignal"></a>
Inject one or more channel-wide periodic signals into a real-world data set. Fake periodic signals are assumed to have a boxcar single pulse shape with uniform pulse amplitude distribution.<br>

Execution syntax from repo base folder:
```
python executables/inject_signal.py -i config/inject_signal.cfg | tee <Log file>
```

---
6. ``simulate-data`` <a name="simulate-data"></a>
Build an artificial filterbank file with one or more channel-wide periodic signals superposed on a Gaussian white noise background. Injected signals have boxcar single-pulse shapes and a constant pulse amplitude.

Reads simulation parameters from a YAML config file. Key configuration sections:
- `output`: basename and output directory for the generated `.fil` file
- `simulation_properties`: number of samples and channels, sampling time, channel bandwidth, first channel frequency, and random seed
- `periodic_signal_injection`: lists of channels, periods (s), duty cycles, pulse S/N values, and initial phases for each injected signal
- `optional_header_parameters`: metadata fields (e.g., source name, start MJD) written into the filterbank header

Execution syntax:
```bash
simulate-data --config config/simulate_data.yaml
```

## Troubleshooting <a name="troubleshooting"></a>
Please submit an issue to voice any problems or requests.

Improvements to the code are always welcome. Check out [CONTRIBUTING.md](https://github.com/UCBerkeleySETI/blipss/blob/main/CONTRIBUTING.md) for best practices on how to contribute to this repository.
