"""Package-wide constants shared across blipss modules"""

# Default floating-point tolerance for harmonic period matching
DEFAULT_EPSILON_HARMONIC: float = 1.0e-3

# Single-character harmonic classification flags assigned to periodicity candidates.
FUNDAMENTAL_FLAG: str = "F"
HARMONIC_FLAG: str = "H"
SUBHARMONIC_FLAG: str = "S"

# Decimal places applied when rounding output period and S/N candidate arrays
CANDIDATE_DECIMAL_PRECISION: int = 5

# Plotting
LABEL_FONTSIZE: int = 16
TICK_LABELSIZE: int = 14
TICK_LENGTH: int = 7
SCATTER_MARKER_SIZE: int = 38

# Column headers for FFA candidate detection output CSV files.
FFA_CANDIDATE_CSV_COLUMNS: list[str] = [
    "Channel",
    "Radio frequency (MHz)",
    "Bins",
    "Best width",
    "Period (s)",
    "S/N",
    "Harmonic flag",
]

# Supported filterbank file extensions for reading and writing.
FILTERBANK_EXTENSIONS: frozenset[str] = frozenset({".fil", ".h5"})

# Default S/N thresholds for cross-file candidate comparison, keyed by pointing type.
DEFAULT_ON_SNR_CUTOFF: float = 7.5
DEFAULT_OFF_SNR_CUTOFF: float = 6.0

# Default period clustering radius (s) for cross-file candidate comparison.
DEFAULT_CLUSTER_RADIUS_SECONDS: float = 1.0e-3

# Column headers for cross-file candidate comparison output CSV files.
COMPARE_CANDS_CSV_COLUMNS: list[str] = [
    "Channel",
    "Radio frequency (MHz)",
    "Bins",
    "Best width",
    "Period (s)",
    "S/N",
    "Code",
]

# Sigproc header constants for simulated filterbank data products.
# Reference: https://sigproc.sourceforge.net/sigproc.pdf
# machine_id = 0: FAKE / unspecified backend (sigproc convention)
# telescope_id = 0: FAKE / unspecified telescope (sigproc convention)
# data_type = 1: filterbank (time-frequency) data (sigproc convention; 2 = time series)
# nbits = 32: samples stored as 32-bit floats
# nifs = 1: single polarisation / flux density feed
SIGPROC_MACHINE_ID: int = 0
SIGPROC_TELESCOPE_ID: int = 0
SIGPROC_DATA_TYPE: int = 1
SIGPROC_N_BITS: int = 32
SIGPROC_N_IFS: int = 1
