# RNA Folding Prediction

This module implements a hybrid RNA 3D structure prediction pipeline combining **Boltz-1** and **DRfold2**.

## Structure

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
