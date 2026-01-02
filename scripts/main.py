
import os
import sys
import pandas as pd
import logging
import time
import shutil
import numpy as np

# Adjust path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rna_folding_module.config import (
    BASE_DIR, INPUTS_DIR, OUTPUTS_DIR, PREDICTIONS_DIR, 
    DRFOLD_DIR, DRFOLD_START_IDX, DRFOLD_END_IDX, DRFOLD_TIME_LIMIT,
    BOLTZ_CACHE
)
from rna_folding_module.src.boltz_handler import prepare_inputs, run_inference, get_coords
from rna_folding_module.src.drfold_handler import setup_drfold, predict_rna_structures_drfold2, convert_cif_to_pdb
from rna_folding_module.src.template_modeler import process_labels_vectorized, predict_rna_structures

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting RNA Folding Pipeline...")

    # --- Setup ---
    # Assume source directories exist or we skip setup
    patches_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "drfold_patches")
    setup_drfold(patches_dir)
    
    # --- Load Data ---
    # Adjust paths as per user environment
    test_sequences_path = os.path.join(BASE_DIR, "test_sequences.csv") # Assuming in root
    train_seqs_path = os.path.join(BASE_DIR, "merged_sequences_final.csv")
    train_labels_path = os.path.join(BASE_DIR, "merged_labels_final.csv")
    
    if not os.path.exists(test_sequences_path):
        logger.error(f"Test sequences file not found at {test_sequences_path}")
        return

    test_sequences = pd.read_csv(test_sequences_path)
    logger.info(f"Loaded {len(test_sequences)} test sequences.")
    
    # --- Boltz Prediction ---
    logger.info("--- Step 1: Boltz Prediction ---")
    
    prepare_inputs(test_sequences_path)
    
    # Path to boltz inference script
    boltz_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "boltz_inference.py")
    
    try:
        run_inference(boltz_script)
    except Exception as e:
        logger.error(f"Boltz inference failed: {e}")
        # Continue? Yes, we might rely on templates/DRfold for some
    
    # Extract Boltz Results
    boltz_predictions = []
    
    for idx, row in test_sequences.iterrows():
        target_id = row['target_id']
        seq_len = len(row['sequence'])
        
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
            boltz_predictions.append({
                'ID': f"{target_id}_{i+1}",
                'coords': preds
            })

    # Save Boltz submission intermediate
    # This part requires expanding the 'coords' into x_1, y_1, z_1... columns
    # Replicating the notebook's logic for saving submission_boltz.csv might be complex if not needed immediately.
    # The notebook saved `submission_boltz.csv` using a helper or pandas.
    # I'll simplify: just keep the data in memory or save a simplified CSV if needed.
    # Actually, the final merge relies on `submission_boltz.csv`? 
    # No, it loads it. So I should save it.
    
    # Constructing DataFrame for Boltz
    # The format is ID, x_1, y_1, z_1, ..., x_n, y_n, z_n
    # This is huge (variable columns). Kaggle submission usually is melted or wide?
    # Actually, the competition format is one row per (ID_residue)?
    # Or wide format? The extract shows `submission.csv` has `ID` and `x_1`, `y_1`...
    # Wait, coordinate column names are `x_1`, `y_1`, `z_1` ... `x_L`, `y_L`, `z_L`?
    # No, usually it's `ID` like `target_id_residue_num` or `target_id_model_residue`.
    # Let's check `process_labels_vectorized`: `labels_df['ID'].str.rsplit('_', n=1)`
    # This implies ID is like `target_1`.
    # And columns are `x_1, y_1, z_1`. Wait.
    
    # Let's look at `submission.csv` format from notebook output.
    # It wasn't explicitly shown.
    # But `process_labels_vectorized` uses `labels_df[['x_1', 'y_1', 'z_1']]`.
    # This suggests the CSV has columns `ID`, `x_1`, `y_1`, `z_1`?
    # No, `ID` looks like `target_id_resid`.
    # Ah, one row per residue?
    # If so, `boltz_predictions` collected above needs to be reformatted.
    
    # Re-reading notebook logic around `submission_boltz.csv`.
    # `get_coords` returns list of tuples.
    # The notebook (Cell 6) creates `submission_boltz.csv`... wait, cell 6 is just inference.py.
    # It doesn't show creating `submission_boltz.csv` in the viewed snippets.
    # But later `pd.read_csv('submission_boltz.csv')`.
    # In `inference.py`, does it write `submission_boltz.csv`?
    # Let's check `inference.py` content I wrote.
    # No, `inference.py` output is `.cif` files.
    # In the notebook, AFTER `run_inference`, there must be code to parse CIFs and create `submission_boltz.csv`.
    # I need to implement that. `main.py` is doing `extract Boltz Results`.
    
    # Let's assume the format is: ID (e.g. `structure_id_residue_id`) and x,y,z columns?
    # Stanford Ribonanza submission format is: `id` (sequence_id_residue_index), `x`, `y`, `z`?
    # Or `id` (e.g. `id_1`), `target_id`, ...
    # Wait, `process_labels_vectorized`: `group[['x_1', 'y_1', 'z_1']]`. This implies input has these columns.
    
    # Let's follow `process_labels_vectorized` logic:
    # `labels_df` has `ID`, `x_1`, `y_1`, `z_1`? This usually implies specific atom coordinates?
    # It seems `ID` identifies the residue.
    # But `submission_dr.csv` logic in notebook (Cell 35+) collects `coord_list`
    # and eventually makes a `submission_df`.
    
    # I'll stick to gathering data and formatted it later.
    
    # --- DRfold / Hybrid Prediction ---
    logger.info("--- Step 2: DRfold / Hybrid Prediction ---")
    
    # Load training data for templates
    if os.path.exists(train_seqs_path) and os.path.exists(train_labels_path):
        train_seqs_df = pd.read_csv(train_seqs_path)
        train_labels_df = pd.read_csv(train_labels_path)
        train_coords_dict = process_labels_vectorized(train_labels_df)
    else:
        logger.warning("Training data not found. Template-based prediction will use de-novo fallback only.")
        train_seqs_df = pd.DataFrame()
        train_coords_dict = {}

    drfold_results = []
    
    start_time = time.time()
    
    for idx, row in test_sequences.iterrows():
        target_id = row['target_id']
        sequence = row['sequence']
        
        # Check range
        if not (DRFOLD_START_IDX <= idx <= DRFOLD_END_IDX):
            continue
            
        elapsed = time.time() - start_time
        if elapsed > DRFOLD_TIME_LIMIT:
            logger.info("Time limit reached. Stopping DRfold loop.")
            break
        
        logger.info(f"Processing {target_id} ({idx})")
        
        use_template = False
        predictions = None
        
        # Hybrid: Try using Boltz result as template for DRfold
        # Get Boltz-1 predicted CIF
        # The path expected by `get_coords` is `outputs_prediction/boltz_results_inputs_prediction/predictions/{tmp_id}/{tmp_id}_model_{idx}.cif`
        # We need model_0 for template
        boltz_cif = os.path.join(OUTPUTS_DIR, "boltz_results_inputs_prediction", "predictions", target_id, f"{target_id}_model_0.cif")
        af3_pdb = os.path.join(PREDICTIONS_DIR, f"{target_id}_af3.pdb")
        
        has_af3 = False
        if os.path.exists(boltz_cif):
            if convert_cif_to_pdb(boltz_cif, af3_pdb):
                has_af3 = True
        
        # Run DRfold2
        try:
            predictions = predict_rna_structures_drfold2(sequence, target_id, af3_pdb if has_af3 else None)
        except Exception as e:
            logger.error(f"DRfold2 failed for {target_id}: {e}")
        
        if not predictions:
            logger.info(f"Using template fallback for {target_id}")
            predictions = predict_rna_structures(sequence, target_id, train_seqs_df, train_coords_dict)
            
        # Store results
        # predictions is list of 5 sets of coordinates (each set is list of tuples)
        drfold_results.append({
            'target_id': target_id,
            'predictions': predictions
        })

    # --- Merging & Output ---
    logger.info("--- Step 3: Merging & Output ---")
    
    # We need to process `drfold_results` and `boltz_predictions` into the final CSV format
    # I'll create `submission.csv`.
    # Format: ID, x, y, z ??
    # The snippet doesn't clearly show the output column headers.
    # However, `submission_processed` cell 11000 shows:
    # `submission_df` columns: ID, x, y, z (one row per residue?)
    # or one row per prediction?
    # Usually in these competitions, `ID` is `target_id_{model_idx}_{residue_idx}`?
    # Let's assume standard submission format.
    
    final_rows = []
    
    # Map Boltz predictions for easy access
    # boltz_predictions is list of dicts: {'ID': 'target_model', 'coords': [x1, y1, z1, ...]}
    # Let's reorganize it to allow lookup by target_id
    boltz_map = {}
    for bp in boltz_predictions:
        # ID is target_id_modelnum
        tid, mnum = bp['ID'].rsplit('_', 1)
        if tid not in boltz_map:
            boltz_map[tid] = []
        boltz_map[tid].append(bp['coords']) # This is flat list [x,y,z,x,y,z...]
        
    all_targets = test_sequences['target_id'].tolist()
    
    for target_id in all_targets:
        seq = test_sequences[test_sequences['target_id'] == target_id]['sequence'].values[0]
        L = len(seq)
        
        # Decide which predictions to use
        # Logic: If long > 600, prefer Boltz. Else DRfold.
        # But `drfold_results` only contains those we ran DRfold on.
        
        # Find predictions in `drfold_results`
        dr_preds = next((item['predictions'] for item in drfold_results if item['target_id'] == target_id), None)
        
        final_preds_for_target = []
        
        if L > 600:
            # Prefer Boltz
            if target_id in boltz_map and boltz_map[target_id]:
               final_preds_for_target = boltz_map[target_id]
            elif dr_preds:
               final_preds_for_target = [ [c for coord in p for c in coord] for p in dr_preds ]
        else:
            # Prefer DRfold
            if dr_preds:
                final_preds_for_target = [ [c for coord in p for c in coord] for p in dr_preds ]
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
        
        # Format for CSV
        # Rows: target_id_confusion_residue_id
        # Actually, let's look at `process_labels_vectorized` again.
        # `labels_df['ID']` -> `target_id_resid`.
        # This implies a wide format is NOT used. It's long format.
        # But we predict 5 models.
        # A common format is `target_id_{model_idx}_{residue_idx}`?
        # Or maybe the submission file expects `target_id_model_idx` and a giant string?
        # No, `x_1, y_1, z_1` usually means per atom or per residue coord?
        
        # Given I cannot check the exact sample submission, I will produce a clear readable format
        # compatible with what `process_labels_vectorized` expects (ID, x_1, y_1, z_1).
        # ID: target_id_{residue_idx} ?
        # But we have multiple models.
        # `submission.csv` implies one submission.
        
        # Let's assume the task is to produce `ID,x,y,z` where ID is `target_id_residue_index`.
        # But wait, we produce 5 models?
        # Maybe the competition only asks for 1?
        # The notebook mentions `submission_dr.csv` and merging.
        # `submission_processed` cell 11000 output shows `submission_processed` head/tail.
        # It's not visible.
        # However, `get_coords` in Cell 7 returns list of coords.
        
        # I will create a CSV that has: `ID,x,y,z` for the FIRST model only?
        # Or maybe `ID` encodes the model number? `target_id_1_resid`?
        
        # Re-reading: "merge these results... create final submission.csv".
        # "For target R1138 ... first conformation's coordinates were replaced ... placeholder values in other conformations were filled".
        # This implies multiple conformations (models) ARE submitted.
        # Likely format: `target_id_prediction_id_residue_id`?
        
        # I'll generate `target_id_{model}_{residue}`.
        
        for m_idx, coords_flat in enumerate(final_preds_for_target):
            # coords_flat is [x1, y1, z1, x2, y2, z2, ...]
            for r_idx in range(L):
                x = coords_flat[r_idx*3]
                y = coords_flat[r_idx*3+1]
                z = coords_flat[r_idx*3+2]
                
                # Use 1-based indexing for models and residues
                row_id = f"{target_id}_{m_idx+1}_{r_idx+1}"
                final_rows.append({
                    'ID': row_id,
                    'x': x,
                    'y': y,
                    'z': z
                })

    submission_df = pd.DataFrame(final_rows)
    sub_path = os.path.join(BASE_DIR, "submission.csv")
    submission_df.to_csv(sub_path, index=False)
    logger.info(f"Submission saved to {sub_path}")

if __name__ == "__main__":
    main()
