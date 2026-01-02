
import os

# Paths
BASE_DIR = os.getcwd()
INPUTS_DIR = os.path.join(BASE_DIR, "inputs_prediction")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs_prediction")
PREDICTIONS_DIR = os.path.join(BASE_DIR, "predictions_dir")
FASTA_DIR = os.path.join(BASE_DIR, "fasta_dir")
DRFOLD_DIR = os.path.join(BASE_DIR, "DRfold2")

# Boltz settings
BOLTZ_CACHE = os.path.expanduser("~/.boltz")

# DRfold settings
# INDICES for processing range
DRFOLD_START_IDX = 0
DRFOLD_END_IDX = 1000 

DRFOLD_TIME_LIMIT = 30000 # seconds

