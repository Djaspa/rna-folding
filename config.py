from pathlib import Path

# Paths
BASE_DIR = Path.cwd()
INPUTS_DIR = BASE_DIR / "inputs_prediction"
OUTPUTS_DIR = BASE_DIR / "outputs_prediction"
PREDICTIONS_DIR = BASE_DIR / "predictions_dir"
FASTA_DIR = BASE_DIR / "fasta_dir"
DRFOLD_DIR = BASE_DIR / "DRfold2"

# Boltz settings
BOLTZ_CACHE = Path.home() / ".boltz"

# DRfold settings
# INDICES for processing range
DRFOLD_START_IDX = 0
DRFOLD_END_IDX = 1000

DRFOLD_TIME_LIMIT = 30000  # seconds
