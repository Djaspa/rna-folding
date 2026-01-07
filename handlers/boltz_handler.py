import logging
import subprocess
from pathlib import Path

import pandas as pd
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def get_outputs_dir(cfg: DictConfig, root_dir: Path = None) -> Path:
    """Get the outputs directory from config."""
    if root_dir is None:
        root_dir = Path.cwd()
    return root_dir / cfg.paths.outputs_dir


def get_inputs_dir(cfg: DictConfig, root_dir: Path = None) -> Path:
    """Get the inputs directory from config."""
    if root_dir is None:
        root_dir = Path.cwd()
    return root_dir / cfg.paths.inputs_dir


def prepare_inputs(test_sequences_csv, cfg: DictConfig = None):
    """
    Generates YAML input files for Boltz inference from the test sequences CSV.
    """
    root_dir = Path.cwd()
    if cfg is not None:
        inputs_dir = get_inputs_dir(cfg, root_dir)
    else:
        inputs_dir = root_dir / "inputs_prediction"

    inputs_dir.mkdir(parents=True, exist_ok=True)

    sub_file = pd.read_csv(test_sequences_csv)
    names = sub_file["target_id"].tolist()
    sequences = sub_file["sequence"].tolist()

    for tmp_id, tmp_sequence in zip(names, sequences):
        with open(inputs_dir / f"{tmp_id}.yaml", "w") as f:
            f.write("constraints: []\n")
            f.write("sequences:\n")
            f.write("- rna:\n")
            f.write("    id:\n")
            f.write("    - A1\n")
            f.write(f"    sequence: {tmp_sequence}")


def run_inference(inference_script_path, cfg: DictConfig = None):
    """
    Runs the Boltz inference script.
    """
    logger.info("Starting Boltz inference...")

    cmd = ["python", str(inference_script_path)]

    cwd = Path.cwd()

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    logger.info(f"Command output: {result.stdout}")
    if result.returncode != 0:
        logger.error(f"Command error: {result.stderr}")
        raise RuntimeError("Boltz inference failed")

    logger.info("Boltz inference completed.")


def get_coords(tmp_id, idx, cfg: DictConfig = None):
    """
    Extracts C1' coordinates from the generated CIF file.
    """
    root_dir = Path.cwd()
    if cfg is not None:
        outputs_dir = get_outputs_dir(cfg, root_dir)
    else:
        outputs_dir = root_dir / "outputs_prediction"

    cif_dir = outputs_dir / "boltz_results_inputs_prediction" / "predictions" / tmp_id
    cif_file = cif_dir / f"{tmp_id}_model_{idx}.cif"

    if not cif_file.exists():
        logger.warning(f"CIF file not found: {cif_file}")
        return None

    mmcif_dict = MMCIF2Dict(str(cif_file))

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
