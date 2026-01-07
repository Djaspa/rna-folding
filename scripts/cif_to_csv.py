#!/usr/bin/env python3
"""CIF to CSV data preparation script for RNA folding datasets.

This script extracts RNA sequences and C1' atom coordinates from CIF (Crystallographic
Information File) files and exports them to CSV files for use in RNA folding models.

The script processes .cif files from a specified directory, extracts:
1. RNA sequences (unique per structure)
2. C1' atom coordinates for each nucleotide

Output files:
- rna_sequences.csv: Contains target_id and sequence columns
- rna_coordinates.csv: Contains ID, resname, resid, x_1, y_1, z_1 columns
"""
from pathlib import Path

import fire
import pandas as pd
from Bio.PDB import MMCIFParser
from tqdm import tqdm


def extract_rna_data_from_cif(cif_file_path: Path) -> tuple[list[dict], list[dict]]:
    """Extract unique RNA sequences and C1' coordinates from a CIF file.

    Args:
        cif_file_path: Path to the CIF file to process.

    Returns:
        Tuple of (sequences_data, coordinates_data) where:
            - sequences_data: List of dicts with 'target_id' and 'sequence' keys
            - coordinates_data: List of dicts with 'ID', 'resname', 'resid',
              'x_1', 'y_1', 'z_1' keys
    """
    parser = MMCIFParser(QUIET=True)

    try:
        structure = parser.get_structure("structure", str(cif_file_path))
        pdb_id = cif_file_path.stem.upper()

        sequences_data = []
        coordinates_data = []
        seen_sequences = set()  # Track unique sequences

        for model in structure:
            for chain in model:
                chain_id = chain.id
                target_id = f"{pdb_id}_{chain_id}"

                # Check if chain contains RNA residues
                rna_residues = []
                for residue in chain:
                    if residue.get_resname() in ["A", "U", "G", "C"]:  # RNA nucleotides
                        rna_residues.append(residue)

                if rna_residues:  # Only process if RNA residues found
                    # Build sequence
                    sequence = "".join([res.get_resname() for res in rna_residues])

                    # Only add if sequence is unique
                    if sequence not in seen_sequences:
                        seen_sequences.add(sequence)
                        sequences_data.append(
                            {"target_id": target_id, "sequence": sequence}
                        )

                        # Extract C1' coordinates for this unique sequence
                        for i, residue in enumerate(rna_residues, 1):
                            if "C1'" in residue:
                                atom = residue["C1'"]
                                coordinates_data.append(
                                    {
                                        "ID": f"{target_id}_{i}",
                                        "resname": residue.get_resname(),
                                        "resid": i,
                                        "x_1": atom.coord[0],
                                        "y_1": atom.coord[1],
                                        "z_1": atom.coord[2],
                                    }
                                )

        return sequences_data, coordinates_data

    except Exception as e:
        print(f"Error processing {cif_file_path}: {e}")
        return [], []


def process_all_cif_files(
    cif_dir: Path, show_progress: bool = True
) -> tuple[list[dict], list[dict]]:
    """Process all CIF files in a directory and extract RNA data.

    Args:
        cif_dir: Path to the directory containing CIF files.
        show_progress: Whether to show a progress bar.

    Returns:
        Tuple of (all_sequences, all_coordinates) lists.
    """
    cif_files = list(cif_dir.glob("*.cif"))

    if not cif_files:
        raise FileNotFoundError(f"No CIF files found in {cif_dir}")

    all_sequences = []
    all_coordinates = []

    print(f"Processing {len(cif_files)} CIF files...")

    iterator = tqdm(cif_files) if show_progress else cif_files
    for cif_file in iterator:
        sequences, coordinates = extract_rna_data_from_cif(cif_file)
        all_sequences.extend(sequences)
        all_coordinates.extend(coordinates)

    return all_sequences, all_coordinates


def main(
    cif_dir: str = "data/stanford-rna-3d-folding/PDB_RNA",
    output_dir: str = "data/rna-cif-to-csv",
    no_progress: bool = False,
) -> None:
    """Main entry point for CIF to CSV conversion.

    Args:
        cif_dir: Path to the directory containing CIF files.
        output_dir: Directory to save the output CSV files.
        no_progress: If True, disable progress bar.
    """
    cif_path = Path(cif_dir)
    output_path = Path(output_dir)

    if not cif_path.exists():
        raise FileNotFoundError(f"CIF directory not found: {cif_path}")

    print("=" * 60)
    print("CIF to CSV Extraction")
    print("=" * 60)
    print(f"Input directory: {cif_path}")
    print(f"Output directory: {output_path}")

    # Process all CIF files
    all_sequences, all_coordinates = process_all_cif_files(
        cif_path, show_progress=not no_progress
    )

    print("\n Extraction summary:")
    print(f"  Total unique RNA sequences: {len(all_sequences)}")
    print(f"  Total coordinate entries: {len(all_coordinates)}")

    # Create DataFrames
    sequences_df = pd.DataFrame(all_sequences)
    coordinates_df = pd.DataFrame(all_coordinates)

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Save to CSV files
    seq_path = output_path / "rna_sequences.csv"
    coord_path = output_path / "rna_coordinates.csv"

    sequences_df.to_csv(seq_path, index=False)
    coordinates_df.to_csv(coord_path, index=False)

    print("\n Saved files:")
    print(f"  {seq_path}: {sequences_df.shape}")
    print(f"  {coord_path}: {coordinates_df.shape}")

    print("\n" + "=" * 60)
    print("CIF to CSV extraction complete!")
    print("=" * 60)


if __name__ == "__main__":
    fire.Fire(main)
