
import os
import shutil
import subprocess
import logging
from Bio.PDB import MMCIFParser, PDBIO, Select
from ..config import DRFOLD_DIR, FASTA_DIR, PREDICTIONS_DIR

logger = logging.getLogger(__name__)

class RNAAtomSelect(Select):
    """Select only RNA backbone and base atoms needed by DRfold2"""
    def accept_atom(self, atom):
        # Check if this is an atom DRfold2 needs
        atom_name = atom.name
        residue = atom.get_parent()
        resname = residue.get_resname()
        
        # Main backbone atoms needed by DRfold2
        if atom_name in ["P", "C4'"]:
            return True
        
        # For purines (A, G) we need N9, for pyrimidines (C, U) we need N1
        if atom_name == "N9" and resname in ["A", "G"]:
            return True
        if atom_name == "N1" and resname in ["C", "U"]:
            return True
        
        return False

def convert_cif_to_pdb(cif_file, pdb_file):
    """Convert mmCIF file to PDB format, fixing chain IDs and keeping only needed atoms."""
    try:
        # Parse the mmCIF file
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure('', cif_file)
        
        # Fix chain IDs (map multi-character IDs like 'A1' to single characters)
        for model in structure:
            for chain in model:
                if len(chain.id) > 1:
                    # Just take the first character of the chain ID
                    chain.id = chain.id[0]
        
        # Write to PDB format, selecting only atoms needed by DRfold2
        io = PDBIO()
        io.set_structure(structure)
        io.save(pdb_file, RNAAtomSelect())
        return True
    except Exception as e:
        logger.error(f"Error converting {cif_file} to PDB: {str(e)}")
        
        # Attempt alternative approach if primary method fails
        try:
            parser = MMCIFParser(QUIET=True)
            structure = parser.get_structure('', cif_file)
            
            # Create a new PDB file manually
            with open(pdb_file, 'w') as f:
                atom_num = 1
                
                for model in structure:
                    for chain in model:
                        chain_id = 'A'  # Use 'A' regardless of original ID
                        
                        for residue in chain:
                            resname = residue.get_resname()
                            resnum = residue.id[1]
                            
                            # Select atoms based on residue type
                            needed_atoms = ["P", "C4'"]
                            if resname in ["A", "G"]:
                                needed_atoms.append("N9")
                            else:  # C, U
                                needed_atoms.append("N1")
                                
                            for atom_name in needed_atoms:
                                if atom_name in residue:
                                    atom = residue[atom_name]
                                    x, y, z = atom.coord
                                    
                                    # Format as PDB ATOM line
                                    line = f"ATOM  {atom_num:5d} {atom_name:<4s} {resname:3s} {chain_id:1s}{resnum:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {atom.element:>2s}  \n"
                                    f.write(line)
                                    atom_num += 1
                
                f.write("END\n")
            
            return True
            
        except Exception as e2:
            logger.error(f"Alternative method also failed: {str(e2)}")
            return False

def setup_drfold(patches_dir):
    """
    Sets up the DRfold2 environment: copies necessary files and applies patches.
    """
    if not os.path.exists(DRFOLD_DIR):
        # In a real scenario, we might clone it. For now, we assume it exists
        # or we log a warning.
        logger.warning(f"DRfold directory {DRFOLD_DIR} does not exist. Please ensure it is present.")
    
    # Apply patches
    logger.info(f"Applying patches from {patches_dir} to {DRFOLD_DIR}")
    
    # Map patch files to their destinations in DRfold2
    # The structure in notebooks suggests:
    # patches/Optimization.py -> DRfold2/PotentialFold/Optimization.py
    # patches/Selection.py -> DRfold2/PotentialFold/Selection.py
    # patches/Cubic.py -> DRfold2/PotentialFold/Cubic.py
    # patches/operations.py -> DRfold2/PotentialFold/operations.py
    # patches/cfg_for_folding.json -> DRfold2/cfg_for_folding.json
    # patches/cfg_for_selection.json -> DRfold2/cfg_for_selection.json
    
    patch_map = {
        "Optimization.py": "PotentialFold/Optimization.py",
        "Selection.py": "PotentialFold/Selection.py",
        "Cubic.py": "PotentialFold/Cubic.py",
        "operations.py": "PotentialFold/operations.py",
        "cfg_for_folding.json": "cfg_for_folding.json",
        "cfg_for_selection.json": "cfg_for_selection.json",
        "DRfold_infer.py": "DRfold_infer.py"
    }

    for patch_file, dest_rel_path in patch_map.items():
        src = os.path.join(patches_dir, patch_file)
        dest = os.path.join(DRFOLD_DIR, dest_rel_path)
        
        if os.path.exists(src):
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy2(src, dest)
            logger.info(f"Copied {src} to {dest}")
        else:
            logger.warning(f"Patch file {src} not found")

    # Compile Arena
    arena_src = os.path.join(DRFOLD_DIR, "Arena", "Arena.cpp")
    arena_exe = os.path.join(DRFOLD_DIR, "Arena", "Arena")
    if os.path.exists(arena_src):
        logger.info("Compiling Arena...")
        subprocess.run(["g++", "-O3", arena_src, "-o", arena_exe], check=True)
        logger.info("Arena compiled.")
    else:
        logger.warning(f"Arena source {arena_src} not found.")


def predict_rna_structures_drfold2(sequence, target_id, af3_pdb=None, is_submission_mode=False):
    """
    Use DRfold2 to predict RNA structures with proper output capture
    """
    
    # Create FASTA file for this sequence
    if not os.path.exists(FASTA_DIR):
        os.makedirs(FASTA_DIR)
        
    fasta_path = os.path.join(FASTA_DIR, f"{target_id}.fasta")
    with open(fasta_path, "w") as f:
        f.write(f">{target_id}\n{sequence}\n")
    
    # Run DRfold2 with proper output capture
    if not os.path.exists(PREDICTIONS_DIR):
        os.makedirs(PREDICTIONS_DIR)
        
    output_dir = os.path.join(PREDICTIONS_DIR, target_id)
    
    # Build command with optional AF3 integration
    # We assume 'scripts/drfold_infer.py' acts as the entry point or we call the one in DRfold2 dir?
    # The notebook writes to /kaggle/working/DRfold2/DRfold_infer.py
    # And calls `python /kaggle/working/DRfold2/DRfold_infer.py ...`
    # We should probably use that script location if we patched it/copied it there.
    # We will assume we copy our `scripts/drfold_infer.py` to `DRfold2/DRfold_infer.py` during setup?
    # Or just call it from `scripts/` but it might depend on CWD.
    # The script uses `exp_dir = os.path.dirname(os.path.abspath(__file__))` to find other files.
    # So it MUST reside in the `DRfold2` directory structure.
    
    drfold_script = os.path.join(DRFOLD_DIR, "DRfold_infer.py")
    
    cmd = f"python {drfold_script} {fasta_path} {output_dir} 1"
    if af3_pdb and os.path.exists(af3_pdb):
        cmd += f" --af3 {af3_pdb}"
        logger.info(f"Using AlphaFold3 structure from: {af3_pdb}")
    elif af3_pdb:
        logger.warning(f"AlphaFold3 file not found: {af3_pdb}")
    
    logger.info(f"Running command: {cmd}")
    process = subprocess.Popen(
        cmd, 
        shell=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    # Print output in real-time
    for line in iter(process.stdout.readline, ''):
        line = line.strip()
        if line:
            print(line) # Print to stdout so user can see progress
    
    # Get return code and check success
    return_code = process.wait()
    if return_code != 0:
        logger.error(f"DRfold2 failed with return code {return_code}")
        return None
    
    # Clean up FASTA file to save space
    if os.path.exists(fasta_path):
        os.remove(fasta_path)
    
    # Extract coordinates
    relax_dir = os.path.join(output_dir, "relax")
    if not os.path.isdir(relax_dir):
        logger.warning(f"No relax directory found for {target_id}")
        relax_dir = output_dir
    
    # Get up to 5 PDB files
    pdb_files = sorted([f for f in os.listdir(relax_dir) if f.endswith(".pdb")])[:5]
    
    if not pdb_files:
        logger.warning(f"No PDB files found for {target_id}")
        return None
    
    # Parse PDB files to extract C1' coordinates
    predictions = []
    for pdb_file in pdb_files:
        file_path = os.path.join(relax_dir, pdb_file)
        
        # Read PDB file
        coords = []
        with open(file_path, "r") as f:
            residue_map = {}
            for line in f:
                if line.startswith("ATOM") and " C1' " in line:
                    parts = line.split()
                    resid = int(parts[5])  # Residue ID as integer
                    x, y, z = float(parts[6]), float(parts[7]), float(parts[8])
                    residue_map[resid] = (x, y, z)
            
            # Ensure we have coordinates for all residues
            for j in range(1, len(sequence) + 1):
                if j in residue_map:
                    coords.append(residue_map[j])
                else:
                    # If residue not found, use zeros
                    logger.warning(f"Residue {j} not found in {pdb_file} for {target_id}")
                    coords.append((0.0, 0.0, 0.0))
        
        predictions.append(coords)
    
    # Clean up PDB files to save space
    if is_submission_mode and os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    # If we have fewer than 5 predictions, duplicate the last one
    while len(predictions) < 5:
        predictions.append(predictions[-1] if predictions else [(0.0, 0.0, 0.0) for _ in range(len(sequence))])
    
    return predictions[:5]  # Return exactly 5 predictions
