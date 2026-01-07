import logging
import sys
import time
from pathlib import Path

import fire
import pandas as pd

# Adjust path to import modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rna_folding_module.config import (  # noqa: E402
    BASE_DIR,
    DRFOLD_END_IDX,
    DRFOLD_START_IDX,
    DRFOLD_TIME_LIMIT,
    OUTPUTS_DIR,
    PREDICTIONS_DIR,
)
from rna_folding_module.src.boltz_handler import (  # noqa: E402
    get_coords,
    prepare_inputs,
    run_inference,
)
from rna_folding_module.src.drfold_handler import (  # noqa: E402
    convert_cif_to_pdb,
    predict_rna_structures_drfold2,
    setup_drfold,
)
from rna_folding_module.src.template_modeler import (  # noqa: E402
    predict_rna_structures,
    process_labels_vectorized,
)

from dvc_utils import ensure_data_available  # noqa: E402

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main(no_dvc_pull: bool = False):
    """Run RNA Folding Pipeline.

    Args:
        no_dvc_pull: If True, skip automatic DVC pull for data files.
    """
    logger.info("Starting RNA Folding Pipeline...")

    # --- Setup ---
    # Assume source directories exist or we skip setup
    root_dir = Path(__file__).resolve().parent.parent
    patches_dir = root_dir / "drfold_patches"
    setup_drfold(patches_dir)

    # --- Load Data ---
    # Data paths (now in data/ directory, managed by DVC)
    test_sequences_path = BASE_DIR / "test_sequences.csv"
    train_seqs_path = root_dir / "data" / "merged_sequences_final.csv"
    train_labels_path = root_dir / "data" / "merged_labels_final.csv"

    # Ensure training data is available (auto-pull from DVC if needed)
    ensure_data_available(
        [train_seqs_path, train_labels_path],
        auto_pull=not no_dvc_pull,
        project_root=root_dir,
    )

    if not test_sequences_path.exists():
        logger.error(f"Test sequences file not found at {test_sequences_path}")
        return

    test_sequences = pd.read_csv(test_sequences_path)
    logger.info(f"Loaded {len(test_sequences)} test sequences.")

    # --- Boltz Prediction ---
    logger.info("--- Step 1: Boltz Prediction ---")

    prepare_inputs(test_sequences_path)

    # Path to boltz inference script
    boltz_script = Path(__file__).resolve().parent / "boltz_inference.py"

    try:
        run_inference(boltz_script)
    except Exception as e:
        logger.error(f"Boltz inference failed: {e}")

    # Extract Boltz Results
    boltz_predictions = []

    for idx, row in test_sequences.iterrows():
        target_id = row["target_id"]
        seq_len = len(row["sequence"])

        # Get up to 5 conformations
        target_preds = []
        for model_idx in range(5):
            coords = get_coords(target_id, model_idx)
            if coords:
                # Format for submission (flatten list)
                flat_coords = [val for coord in coords for val in coord]
                target_preds.append(flat_coords)
            else:
                # Placeholder
                target_preds.append([0.0] * (seq_len * 3))

        # Add to predictions list
        for i, preds in enumerate(target_preds):
            boltz_predictions.append({"ID": f"{target_id}_{i+1}", "coords": preds})

    # --- DRfold / Hybrid Prediction ---
    logger.info("--- Step 2: DRfold / Hybrid Prediction ---")

    # Load training data for templates
    if train_seqs_path.exists() and train_labels_path.exists():
        train_seqs_df = pd.read_csv(train_seqs_path)
        train_labels_df = pd.read_csv(train_labels_path)
        train_coords_dict = process_labels_vectorized(train_labels_df)
    else:
        logger.warning(
            "Training data not found. Template-based prediction will use de-novo "
            "fallback only."
        )
        train_seqs_df = pd.DataFrame()
        train_coords_dict = {}

    drfold_results = []

    start_time = time.time()

    for idx, row in test_sequences.iterrows():
        target_id = row["target_id"]
        sequence = row["sequence"]

        # Check range
        if not (DRFOLD_START_IDX <= idx <= DRFOLD_END_IDX):
            continue

        elapsed = time.time() - start_time
        if elapsed > DRFOLD_TIME_LIMIT:
            logger.info("Time limit reached. Stopping DRfold loop.")
            break

        logger.info(f"Processing {target_id} ({idx})")

        use_template = False  # noqa: F841
        predictions = None

        # Hybrid: Try using Boltz result as template for DRfold
        # Get Boltz-1 predicted CIF
        # The path expected by `get_coords` is
        # `outputs_prediction/boltz_results_inputs_prediction/predictions/{tmp_id}/{tmp_id}_model_{idx}.cif`
        # We need model_0 for template
        boltz_cif = (
            OUTPUTS_DIR
            / "boltz_results_inputs_prediction"
            / "predictions"
            / target_id
            / f"{target_id}_model_0.cif"
        )
        af3_pdb = PREDICTIONS_DIR / f"{target_id}_af3.pdb"

        has_af3 = False
        if boltz_cif.exists():
            if convert_cif_to_pdb(boltz_cif, af3_pdb):
                has_af3 = True

        # Run DRfold2
        try:
            predictions = predict_rna_structures_drfold2(
                sequence, target_id, af3_pdb if has_af3 else None
            )
        except Exception as e:
            logger.error(f"DRfold2 failed for {target_id}: {e}")

        if not predictions:
            logger.info(f"Using template fallback for {target_id}")
            predictions = predict_rna_structures(
                sequence, target_id, train_seqs_df, train_coords_dict
            )

        # Store results
        # predictions is list of 5 sets of coordinates (each set is list of tuples)
        drfold_results.append({"target_id": target_id, "predictions": predictions})

    # --- Merging & Output ---
    logger.info("--- Step 3: Merging & Output ---")

    final_rows = []

    # Map Boltz predictions for easy access
    # boltz_predictions is list of dicts:
    # {'ID': 'target_model', 'coords': [x1, y1, z1, ...]}
    # Let's reorganize it to allow lookup by target_id
    boltz_map = {}
    for bp in boltz_predictions:
        # ID is target_id_modelnum
        tid, mnum = bp["ID"].rsplit("_", 1)
        if tid not in boltz_map:
            boltz_map[tid] = []
        boltz_map[tid].append(bp["coords"])  # This is flat list [x,y,z,x,y,z...]

    all_targets = test_sequences["target_id"].tolist()

    for target_id in all_targets:
        seq = test_sequences[test_sequences["target_id"] == target_id][
            "sequence"
        ].values[0]
        L = len(seq)

        # Decide which predictions to use
        # Logic: If long > 600, prefer Boltz. Else DRfold.
        # But `drfold_results` only contains those we ran DRfold on.

        # Find predictions in `drfold_results`
        dr_preds = next(
            (
                item["predictions"]
                for item in drfold_results
                if item["target_id"] == target_id
            ),
            None,
        )

        final_preds_for_target = []

        if L > 600:
            # Prefer Boltz
            if target_id in boltz_map and boltz_map[target_id]:
                final_preds_for_target = boltz_map[target_id]
            elif dr_preds:
                final_preds_for_target = [
                    [c for coord in p for c in coord] for p in dr_preds
                ]
        else:
            # Prefer DRfold
            if dr_preds:
                final_preds_for_target = [
                    [c for coord in p for c in coord] for p in dr_preds
                ]
            elif target_id in boltz_map and boltz_map[target_id]:
                final_preds_for_target = boltz_map[target_id]

        # Ensure we have 5 predictions
        while len(final_preds_for_target) < 5:
            # Fill with zeros or duplicate
            if final_preds_for_target:
                final_preds_for_target.append(final_preds_for_target[-1])
            else:
                final_preds_for_target.append([0.0] * (L * 3))

        # Truncate if > 5
        final_preds_for_target = final_preds_for_target[:5]

        for m_idx, coords_flat in enumerate(final_preds_for_target):
            # coords_flat is [x1, y1, z1, x2, y2, z2, ...]
            for r_idx in range(L):
                x = coords_flat[r_idx * 3]
                y = coords_flat[r_idx * 3 + 1]
                z = coords_flat[r_idx * 3 + 2]

                # Use 1-based indexing for models and residues
                row_id = f"{target_id}_{m_idx+1}_{r_idx+1}"
                final_rows.append({"ID": row_id, "x": x, "y": y, "z": z})

    submission_df = pd.DataFrame(final_rows)
    sub_path = BASE_DIR / "submission.csv"
    submission_df.to_csv(sub_path, index=False)
    logger.info(f"Submission saved to {sub_path}")


if __name__ == "__main__":
    fire.Fire(main)
