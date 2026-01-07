import logging
import sys
import time
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

# Adjust path to import modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_processing.dvc_utils import ensure_data_available  # noqa: E402
from handlers.boltz_handler import (  # noqa: E402
    get_coords,
    prepare_inputs,
    run_inference,
)
from handlers.drfold_handler import (  # noqa: E402
    convert_cif_to_pdb,
    predict_rna_structures_drfold2,
    setup_drfold,
)
from handlers.template_modeler import (  # noqa: E402
    predict_rna_structures,
    process_labels_vectorized,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="inference")
def main(cfg: DictConfig):
    """Run RNA Folding Pipeline."""
    logger.info("Starting RNA Folding Pipeline...")

    # Get project root and resolve paths
    root_dir = Path(__file__).resolve().parent.parent

    # --- Setup ---
    patches_dir = root_dir / "drfold_patches"
    setup_drfold(patches_dir, cfg)

    # Resolve paths from config
    base_dir = root_dir / cfg.paths.data_dir
    outputs_dir = root_dir / cfg.paths.outputs_dir
    predictions_dir = root_dir / cfg.paths.predictions_dir

    # --- Load Data ---
    test_sequences_path = base_dir / "test_sequences.csv"
    train_seqs_path = root_dir / "data" / "merged_sequences_final.csv"
    train_labels_path = root_dir / "data" / "merged_labels_final.csv"

    # Ensure training data is available (auto-pull from DVC if needed)
    ensure_data_available(
        [train_seqs_path, train_labels_path],
        auto_pull=cfg.get("no_dvc_pull", False) is False,
        project_root=root_dir,
    )

    if not test_sequences_path.exists():
        logger.error(f"Test sequences file not found at {test_sequences_path}")
        return

    test_sequences = pd.read_csv(test_sequences_path)
    logger.info(f"Loaded {len(test_sequences)} test sequences.")

    # --- Boltz Prediction ---
    logger.info("--- Step 1: Boltz Prediction ---")

    prepare_inputs(test_sequences_path, cfg)

    # Path to boltz inference script
    boltz_script = Path(__file__).resolve().parent / "boltz_inference.py"

    try:
        run_inference(boltz_script, cfg)
    except Exception as e:
        logger.error(f"Boltz inference failed: {e}")

    # Extract Boltz Results
    boltz_predictions = []

    for idx, row in test_sequences.iterrows():
        target_id = row["target_id"]
        seq_len = len(row["sequence"])

        # Get up to 5 conformations
        target_preds = []
        for model_idx in range(cfg.drfold.num_conformations):
            coords = get_coords(target_id, model_idx, cfg)
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

        # Check range from config
        if not (cfg.drfold.start_idx <= idx <= cfg.drfold.end_idx):
            continue

        elapsed = time.time() - start_time
        if elapsed > cfg.drfold.time_limit:
            logger.info("Time limit reached. Stopping DRfold loop.")
            break

        logger.info(f"Processing {target_id} ({idx})")

        use_template = False  # noqa: F841
        predictions = None

        # Hybrid: Try using Boltz result as template for DRfold
        boltz_cif = (
            outputs_dir
            / "boltz_results_inputs_prediction"
            / "predictions"
            / target_id
            / f"{target_id}_model_0.cif"
        )
        af3_pdb = predictions_dir / f"{target_id}_af3.pdb"

        has_af3 = False
        if boltz_cif.exists():
            if convert_cif_to_pdb(boltz_cif, af3_pdb):
                has_af3 = True

        # Run DRfold2
        try:
            predictions = predict_rna_structures_drfold2(
                sequence, target_id, af3_pdb if has_af3 else None, cfg
            )
        except Exception as e:
            logger.error(f"DRfold2 failed for {target_id}: {e}")

        if not predictions:
            logger.info(f"Using template fallback for {target_id}")
            predictions = predict_rna_structures(
                sequence, target_id, train_seqs_df, train_coords_dict
            )

        # Store results
        drfold_results.append({"target_id": target_id, "predictions": predictions})

    # --- Merging & Output ---
    logger.info("--- Step 3: Merging & Output ---")

    final_rows = []

    # Map Boltz predictions for easy access
    boltz_map = {}
    for bp in boltz_predictions:
        tid, mnum = bp["ID"].rsplit("_", 1)
        if tid not in boltz_map:
            boltz_map[tid] = []
        boltz_map[tid].append(bp["coords"])

    all_targets = test_sequences["target_id"].tolist()

    # Sequence length threshold from config
    seq_threshold = cfg.drfold.sequence_length_threshold

    for target_id in all_targets:
        seq = test_sequences[test_sequences["target_id"] == target_id][
            "sequence"
        ].values[0]
        L = len(seq)

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

        if L > seq_threshold:
            # Prefer Boltz for long sequences
            if target_id in boltz_map and boltz_map[target_id]:
                final_preds_for_target = boltz_map[target_id]
            elif dr_preds:
                final_preds_for_target = [
                    [c for coord in p for c in coord] for p in dr_preds
                ]
        else:
            # Prefer DRfold for shorter sequences
            if dr_preds:
                final_preds_for_target = [
                    [c for coord in p for c in coord] for p in dr_preds
                ]
            elif target_id in boltz_map and boltz_map[target_id]:
                final_preds_for_target = boltz_map[target_id]

        # Ensure we have 5 predictions
        while len(final_preds_for_target) < cfg.drfold.num_conformations:
            if final_preds_for_target:
                final_preds_for_target.append(final_preds_for_target[-1])
            else:
                final_preds_for_target.append([0.0] * (L * 3))

        # Truncate if > 5
        final_preds_for_target = final_preds_for_target[: cfg.drfold.num_conformations]

        for m_idx, coords_flat in enumerate(final_preds_for_target):
            for r_idx in range(L):
                x = coords_flat[r_idx * 3]
                y = coords_flat[r_idx * 3 + 1]
                z = coords_flat[r_idx * 3 + 2]

                row_id = f"{target_id}_{m_idx+1}_{r_idx+1}"
                final_rows.append({"ID": row_id, "x": x, "y": y, "z": z})

    submission_df = pd.DataFrame(final_rows)
    sub_path = base_dir / "submission.csv"
    submission_df.to_csv(sub_path, index=False)
    logger.info(f"Submission saved to {sub_path}")


if __name__ == "__main__":
    main()
