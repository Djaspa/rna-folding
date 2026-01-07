# RNA Folding Prediction

This module implements a hybrid RNA 3D structure prediction pipeline combining **Boltz-1** and **DRfold2**.

## Problem Statement

The task is to predict the 3D structure of RNA molecules (C1' atom coordinates for each
nucleotide) from their primary sequence, following the setup of the Kaggle “Stanford RNA 3D
Folding” competition and using the 1st place “hybrid TBM + DRfold2” notebook as the backbone
of the solution. Accurate RNA 3D prediction can improve understanding of RNA function and
enable advances in RNA-based therapeutics and biotechnology.

### Input and Output Data Format

The primary input is a CSV file test_sequences.csv with one row per RNA chain, containing at
least: an ID string and the nucleotide sequence using standard A/U/G/C codes plus, in some
cases, modified bases. Additional template-library inputs are CSVs derived from PDB RNA CIF
files, with columns such as sequence, residue index, residue name, and C1' x, y, z coordinates
for each structural template. The model outputs submission.csv with one row per residue per
target ID; columns include ID, resname, resid, and 15 numeric columns
x_1,y_1,z_1,…,x_5,y_5,z_5 representing the C1' coordinates of up to five predicted models per
sequence, exactly as required by the competition.

### Metrics

The main metric is TM-score, a length-normalized structural similarity measure between 0 and
1, where higher is better and scores above roughly 0.5 typically indicate a generally correct fold
topology. TM-score is used because it is robust to local errors, comparable across different
sequence lengths, and aligned with community practice in CASP and similar structure prediction
benchmarks.

### Validation

Validation will rely on splitting the training data (PDB_RNA-based templates converted to CSV)
into disjoint sets of sequences and structures. To ensure reproducibility, all splits will be driven
by fixed random seeds, version-pinned datasets, and fully scripted preprocessing steps.

### Data

Core data comes from the official competition dataset, including train_sequences.csv,
train_labels (coordinates), and test_sequences.csv.. In addition, the 1st place pipeline
uses extended, preprocessed RNA structural libraries derived from PDB_RNA CIF files,
publicly shared as Kaggle datasets such as rna-all-data, rna-cif-to-csv, DRfold2-repo,
DRfold2-models, rna-wheels, biopython, fairscale-0413, and rna-prediction-boltz. These
external datasets provide: an enlarged template library (~19k+ sequences) with
comprehensive mapping of ~93 modified nucleotides, pre-packaged DRfold2 code and
weights, and auxiliary resources like Boltz-1 potentials;

### Modeling

#### Baseline

A simple baseline is a pure template-based modeling (TBM) system without any deep learning,
closely following the TBM-only notebooks shared by the 1st place team. This baseline: builds a
template library from PDB_RNA-derived CSVs, performs global sequence alignment using
Biopython’s aligner with tuned RNA-specific gap penalties, transfers coordinates for aligned
positions, reconstructs gaps via geometric rules (maintaining typical C1'-C1' distances and
interpolating/backbone-extending where needed), and outputs 3D models without DRfold2
refinement. Earlier TBM-only iterations reached TM-scores around 0.31, while an improved
TBM-only pipeline with better template search, composite similarity scoring, clustering, and
diversity selection achieved about 0.59, so the simpler TBM system provides a well-defined
lower and mid-level reference for measuring hybrid gains.
Main model
The main model is a hybrid pipeline using the 1st place “Hybrid TBM + DRfold2” notebook as
the backbone: a sophisticated TBM front-end combined with DRfold2 for optimization and
refinement. The TBM module performs: (1) global sequence alignment over an extended
template library, (2) composite template ranking using a weighted mixture of global alignment
score, local alignment features, hand-crafted sequence-composition features (e.g., dinucleotide
frequencies), and k-mer similarity, (3) clustering and diversity selection of templates, and (4)
coordinate transfer and backbone gap-filling with confidence-dependent geometric constraints.
The DRfold2 module is used with several enhancements from the winning solution:
double-precision vectorized scoring, GPU-accelerated energy and distance computations
(torch.cdist), PyTorch LBFGS optimization, and integration of external potentials like Boltz-1,
while also improving the model-selection stage that chooses the best conformations among
DRfold2 outputs.

## Project Structure

```
rna_folding/
├── constants.py              # Global constants (vocabulary, structural params)
├── boltz_handler.py          # Boltz-1 input preparation and inference
├── drfold_handler.py         # DRfold2 setup, CIF→PDB conversion, inference
├── template_modeler.py       # Template-based prediction and de novo fallbacks
├── dvc_utils.py              # DVC utility functions for data management
│
├── configs/                  # Hydra configuration files
│   ├── config.yaml           # Main training config entrypoint
│   ├── inference.yaml        # Main inference config entrypoint
│   ├── paths.yaml            # Path configurations
│   ├── training/
│   │   └── default.yaml      # Training hyperparameters
│   ├── model/
│   │   └── default.yaml      # Model architecture parameters
│   ├── data/
│   │   └── default.yaml      # DataModule settings
│   ├── inference/
│   │   ├── drfold.yaml       # DRfold inference settings
│   │   └── boltz.yaml        # Boltz inference settings
│   └── preprocessing/
│       └── default.yaml      # Data preparation settings
│
├── scripts/                  # Executable scripts
│   ├── main.py               # Main orchestration script (uses Hydra)
│   ├── boltz_inference.py    # Standalone Boltz inference (called by handler)
│   ├── cif_to_csv.py         # CIF to CSV extraction script
│   └── prepare_data.py       # Data preparation and merging script
│
├── training/                 # PyTorch Lightning training module
│   ├── data_module.py        # LightningDataModule for RNA datasets
│   ├── lightning_model.py    # LightningModule for model training
│   └── train.py              # Training entry point (uses Hydra)
│
├── drfold_patches/           # Modified DRfold2 source code
│   ├── Optimization.py       # Optimized energy minimization
│   ├── Selection.py          # Model selection logic
│   ├── operations.py         # Vectorized operations
│   ├── Cubic.py              # Cubic spline utilities
│   ├── DRfold_infer.py       # Inference wrapper
│   ├── cfg_for_folding.json  # Folding configuration
│   └── cfg_for_selection.json # Selection configuration
│
└── data/                     # Data directory (create as needed)
    ├── stanford-rna-3d-folding/
    ├── extended-rna/
    └── rna-cif-to-csv/
```

## Data Preparation

The project requires RNA sequence and label data from multiple sources. Use the `prepare_data.py` script to merge these datasets.

### Data Sources

1. **Stanford RNA 3D Folding** (Kaggle competition data)
   - `train_sequences.csv`, `validation_sequences.csv`, `test_sequences.csv`
   - `train_labels.csv`, `validation_labels.csv`

2. **Extended RNA** (additional training data)
   - `train_sequences_v2.csv`, `train_labels_v2.csv`

3. **RNA CIF to CSV** (PDB-derived structural data)
   - `rna_sequences.csv`, `rna_coordinates.csv`

### Extracting RNA Data from CIF Files

If you have PDB CIF files (e.g., from `PDB_RNA/`), use the `cif_to_csv.py` script to extract RNA sequences and C1' coordinates:

```bash
# Extract from default location (data/stanford-rna-3d-folding/PDB_RNA/)
uv run scripts/cif_to_csv.py

# Custom input/output paths
uv run scripts/cif_to_csv.py --cif_dir /path/to/cif/files --output_dir data/rna-cif-to-csv

# Disable progress bar (useful for logging)
uv run scripts/cif_to_csv.py --no_progress
```

This generates:

- `rna_sequences.csv`: Unique RNA sequences with `target_id` and `sequence` columns
- `rna_coordinates.csv`: C1' atom coordinates with `ID`, `resname`, `resid`, `x_1`, `y_1`, `z_1` columns

### Preparing the Data

1. Organize your data files in the expected directory structure:

   ```
   data/
   ├── stanford-rna-3d-folding/
   │   ├── PDB_RNA/             # CIF files (optional, for cif_to_csv.py)
   │   ├── train_sequences.csv
   │   ├── validation_sequences.csv
   │   ├── test_sequences.csv
   │   ├── train_labels.csv
   │   └── validation_labels.csv
   ├── extended-rna/
   │   ├── train_sequences_v2.csv
   │   └── train_labels_v2.csv
   └── rna-cif-to-csv/
       ├── rna_sequences.csv
       └── rna_coordinates.csv
   ```

2. Run the data preparation script:

   ```bash
   # Merge all datasets and export to current directory
   uv run scripts/prepare_data.py --data_dir data --output_dir .

   # Or merge only sequences
   uv run scripts/prepare_data.py --data_dir data --sequences_only

   # Or merge only labels
   uv run scripts/prepare_data.py --data_dir data --labels_only
   ```

3. The script produces:
   - `merged_sequences_final.csv`: ~19,000+ unique RNA sequences
   - `merged_labels_final.csv`: Corresponding 3D coordinates (C1' atoms)

## Prerequisites

- **Python**: >=3.13
- **uv**: Recommended for dependency management.

## Installation

1.  **Install dependencies**:

    ```bash
    uv sync
    ```

2.  **Set up Pre-commit hooks** (for developers):

    ```bash
    uv run pre-commit install
    ```

3.  **Pull data from DVC** (if you have access to the remote):
    ```bash
    uv run dvc pull
    ```

## DVC Data Management

This project uses [DVC (Data Version Control)](https://dvc.org/) to manage large data files. Data is stored on Google Drive and tracked via `.dvc` files in git.

### Tracked Data Files

- `data/merged_sequences_final.csv` - Merged RNA sequences (~10MB)
- `data/merged_labels_final.csv` - Merged 3D coordinates (~544MB)
- `data/stanford-rna-3d-folding/` - Competition data
- `data/extended-rna/` - Extended training data
- `data/rna-cif-to-csv/` - PDB-derived structural data

### Data Commands

```bash
# Pull all data from remote
uv run dvc pull

# Check data status
uv run dvc status

# Push new data to remote (after adding with dvc add)
uv run dvc push
```

### Automatic Data Pull

Training and inference commands automatically pull missing data via DVC. To skip auto-pull:

```bash
# Training
uv run python training/train.py training.no_dvc_pull=true

# Inference
uv run python scripts/main.py no_dvc_pull=true
```

## Usage

### Training

```bash
# Run training (auto-pulls data if missing)
uv run python training/train.py

# Quick validation run
uv run python training/train.py training.fast_dev_run=true

# Custom batch size and epochs
uv run python training/train.py training.batch_size=8 training.epochs=20

# Override model hyperparameters
uv run python training/train.py model.learning_rate=0.01 model.hidden_dim=256

# Show all available configuration options
uv run python training/train.py --help
```

### Inference

1.  Place `DRfold2` repository in the project root (or ensure it's accessible).
2.  Place `test_sequences.csv` in the project root.
3.  Run the inference pipeline:

    ```bash
    uv run python scripts/main.py

    # Override inference settings
    uv run python scripts/main.py drfold.time_limit=60000 drfold.end_idx=500
    ```

## Experiment Tracking (MLflow)

This project uses [MLflow](https://mlflow.org/) for experiment tracking.

### Viewing Experiments

1.  Start the MLflow UI:
    ```bash
    uv run mlflow ui
    ```
2.  Open http://127.0.0.1:5000 in your browser.

### Configuration

Logging is enabled by default. To disable it:

```bash
uv run python training/train.py logging.enabled=false
```

To use a remote tracking server, update `configs/logging/mlflow.yaml` or override via CLI:

```bash
uv run python training/train.py logging.tracking_uri=plots
```

## Configuration

This project uses [Hydra](https://hydra.cc/) for configuration management. All hyperparameters are defined in YAML files under `configs/`.

### Config Structure

| Config                  | Description                                      |
| ----------------------- | ------------------------------------------------ |
| `config.yaml`           | Main training entrypoint                         |
| `inference.yaml`        | Main inference entrypoint                        |
| `paths.yaml`            | Directory paths                                  |
| `training/default.yaml` | Batch size, epochs, checkpoint settings          |
| `model/default.yaml`    | Model architecture (embed_dim, hidden_dim, etc.) |
| `data/default.yaml`     | DataModule settings (val_split, num_workers)     |
| `inference/drfold.yaml` | DRfold settings (time_limit, index ranges)       |
| `inference/boltz.yaml`  | Boltz diffusion parameters                       |

### Overriding Config Values

```bash
# Override single values
uv run python training/train.py training.batch_size=16

# Override nested values
uv run python training/train.py training.checkpoint.save_top_k=5

# Multiple overrides
uv run python training/train.py training.epochs=50 model.learning_rate=0.0001

# Show composed config
uv run python training/train.py --cfg job
```

### Global Constants

Fixed constants that shouldn't be configurable (like RNA vocabulary) are defined in `constants.py`.

## Development

This project uses `pre-commit` to ensure code quality with `black`, `isort`, `flake8`, and `prettier`.

To run checks manually:

```bash
uv run pre-commit run --all-files
```

## Deployment

### ONNX Export

You can export the trained PyTorch model to ONNX format either automatically during training or manually using a script.

#### Automatic Export

Enable the `export_onnx` flag in the training configuration:

```bash
uv run python training/train.py training.export_onnx=true
```

This will export the best model (or final model if checkpointing fails) to `model.onnx` in the project root.

#### Manual Export

Use the export script:

```bash
# Export from a specific checkpoint
uv run python scripts/export_onnx.py --checkpoint_path checkoints/my_model.ckpt --output my_model.onnx

# Export a fresh initialized model (for testing structure)
uv run python scripts/export_onnx.py --output dummy_model.onnx
```

### TensorRT Conversion

To convert an ONNX model to a TensorRT engine, use the conversion script. This requires `trtexec` to be installed and available in your PATH.

```bash
# Basic conversion
uv run python scripts/convert_tensorrt.py --onnx model.onnx --output model.engine

# Enable FP16 precision
uv run python scripts/convert_tensorrt.py --onnx model.onnx --output model.engine --fp16

# Verbose output
uv run python scripts/convert_tensorrt.py --onnx model.onnx --verbose
```

### MLflow Model Serving

Serve your trained model as a REST API using MLflow.

#### 1. Register the Model

First, register a trained checkpoint with MLflow:

```bash
# Register from a checkpoint
uv run python scripts/register_model.py --checkpoint checkpoints/rna-fold-epoch=01-val_loss=nan.ckpt

# Custom model name
uv run python scripts/register_model.py --checkpoint checkpoints/my_model.ckpt --model_name my-rna-model
```

View registered models in the MLflow UI:

```bash
uv run mlflow ui --backend-store-uri plots
```

#### 2. Start the Serving Endpoint

```bash
# Start with default config (port 5001)
uv run python scripts/serve.py

# Custom port
uv run python scripts/serve.py --port 8080

# Specific model version
uv run python scripts/serve.py --model_version 1
```

#### 3. Make Predictions

Send requests to the serving endpoint:

```bash
# Using curl (input: sequence indices where A=0, U=1, G=2, C=3, X=4)
curl -X POST http://127.0.0.1:5001/invocations \
  -H "Content-Type: application/json" \
  -d '{"inputs": [[0, 1, 2, 3, 1, 2, 0, 1, 2, 3]]}'
```

```python
# Using Python requests
import requests

response = requests.post(
    "http://127.0.0.1:5001/invocations",
    json={"inputs": [[0, 1, 2, 3, 1, 2, 0, 1, 2, 3]]}
)
coords = response.json()  # Shape: (batch, seq_len, 3)
```

#### Serving Configuration

Edit `configs/serving/mlflow_serving.yaml` to customize defaults:

```yaml
host: "127.0.0.1"
port: 5001
model_name: "rna-folding-model"
model_version: "latest"
workers: 1
```
