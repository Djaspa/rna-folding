import logging
import subprocess
from pathlib import Path

import pandas as pd
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

from ..config import INPUTS_DIR, OUTPUTS_DIR

logger = logging.getLogger(__name__)


def prepare_inputs(test_sequences_csv):
    """
    Generates YAML input files for Boltz inference from the test sequences CSV.
    """
    INPUTS_DIR.mkdir(parents=True, exist_ok=True)

    sub_file = pd.read_csv(test_sequences_csv)
    names = sub_file["target_id"].tolist()
    sequences = sub_file["sequence"].tolist()

    for tmp_id, tmp_sequence in zip(names, sequences):
        with open(INPUTS_DIR / f"{tmp_id}.yaml", "w") as f:
            f.write("constraints: []\n")
            f.write("sequences:\n")
            f.write("- rna:\n")
            f.write("    id:\n")
            f.write("    - A1\n")
            f.write(f"    sequence: {tmp_sequence}")


def run_inference(inference_script_path):
    """
    Runs the Boltz inference script.
    """
    logger.info("Starting Boltz inference...")
    # Ensure inference script is in the python path or called directly
    # In the notebook it was called as 'python inference.py' assuming it was in CWD.
    # We will pass the absolute path.

    cmd = ["python", str(inference_script_path)]

    # The inference script in the notebook hardcoded paths (e.g. ./inputs_prediction).
    # We might need to ensure the CWD is correct or update the script to take args.
    # The script uses click and __main__ calls predict() with hardcoded args.
    # To avoid modifying the script too much, we'll run it from the base dir where
    # inputs_prediction exists.

    cwd = Path.cwd()  # Should be the project root where inputs_prediction is

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    logger.info(f"Command output: {result.stdout}")
    if result.returncode != 0:
        logger.error(f"Command error: {result.stderr}")
        raise RuntimeError("Boltz inference failed")

    logger.info("Boltz inference completed.")


def get_coords(tmp_id, idx):
    """
    Extracts C1' coordinates from the generated CIF file.
    """
    # The notebook path structure for outputs:
    # outputs_prediction/boltz_results_inputs_prediction/predictions/{tmp_id}/{tmp_id}_model_{idx}.cif

    cif_dir = OUTPUTS_DIR / "boltz_results_inputs_prediction" / "predictions" / tmp_id
    cif_file = cif_dir / f"{tmp_id}_model_{idx}.cif"

    if not cif_file.exists():
        logger.warning(f"CIF file not found: {cif_file}")
        return None

    mmcif_dict = MMCIF2Dict(str(cif_file))
    # entity_poly_seq = mmcif_dict.get("_entity_poly_seq.mon_id", [])
    # sequence = "".join(entity_poly_seq)

    x_coords = mmcif_dict["_atom_site.Cartn_x"]
    y_coords = mmcif_dict["_atom_site.Cartn_y"]
    z_coords = mmcif_dict["_atom_site.Cartn_z"]
    atom_names = mmcif_dict["_atom_site.label_atom_id"]

    c1_coords = []
    for i, atom in enumerate(atom_names):
        if atom == "C1'":
            c1_coords.append(
                (float(x_coords[i]), float(y_coords[i]), float(z_coords[i]))
            )

    return c1_coords
