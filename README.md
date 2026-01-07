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

- **`config.py`**: Configuration parameters.
- **`boltz_handler.py`**: Handles Boltz-1 input preparation and inference.
- **`drfold_handler.py`**: Handles DRfold2 setup, CIF->PDB conversion, and inference.
- **`template_modeler.py`**: Implements template-based prediction and de novo fallbacks.
- **`scripts/`**
  - `main.py`: Main orchestration script.
  - `boltz_inference.py`: Standalone Boltz inference script (called by handler).
- **`drfold_patches/`**: Contains modified source code for DRfold2 (Optimization, Selection, etc.).

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

## Usage

1.  Place `DRfold2` repository in the project root (or ensure it's accessible).
2.  Place input data (`test_sequences.csv`, `merged_sequences_final.csv`, `merged_labels_final.csv`) in the project root.
3.  Run the pipeline:

    ```bash
    uv run scripts/main.py
    ```

## Development

This project uses `pre-commit` to ensure code quality with `black`, `isort`, `flake8`, and `prettier`.

To run checks manually:

```bash
uv run pre-commit run --all-files
```

## Configuration

Modify `config.py` to adjust paths, time limits, and prediction ranges.
