
# RNA Folding Prediction

This module implements a hybrid RNA 3D structure prediction pipeline combining **Boltz-1** and **DRfold2**.

## Structure

- `rna_folding/`
  - `config.py`: Configuration parameters.
  - `boltz_handler.py`: Handles Boltz-1 input preparation and inference.
  - `drfold_handler.py`: Handles DRfold2 setup, CIF->PDB conversion, and inference.
  - `template_modeler.py`: Implements template-based prediction and de novo fallbacks.
  - `scripts/`
    - `main.py`: Main orchestration script.
    - `boltz_inference.py`: Standalone Boltz inference script (called by handler).
  - `drfold_patches/`: Contains modified source code for DRfold2 (Optimization, Selection, etc.).

## Usage

1. Ensure all dependencies are installed.
2. Place `DRfold2` repository in the project root (or ensure it's accessible).
3. Place input data (`test_sequences.csv`, `merged_sequences_final.csv`, `merged_labels_final.csv`) in the project root.
4. Run the pipeline:

```bash
python -m rna_folding.scripts.main
```

## Configuration

Modify `rna_folding/config.py` to adjust paths, time limits, and prediction ranges.
