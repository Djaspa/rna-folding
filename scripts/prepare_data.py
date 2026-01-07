#!/usr/bin/env python3
"""Data preparation script for RNA folding datasets.

This script merges multiple RNA sequence and label datasets from various sources,
removes duplicates, and exports the merged datasets to CSV files.

The script handles three main data sources:
1. Stanford RNA 3D Folding competition data
   (train_sequences.csv, validation_sequences.csv, etc.)
2. Extended RNA data (train_sequences_v2.csv, train_labels_v2.csv)
3. RNA CIF to CSV data (rna_sequences.csv, rna_coordinates.csv)
"""
from pathlib import Path

import fire
import pandas as pd


def load_data(data_dir: Path) -> dict:
    """Load all RNA sequence and label datasets from the data directory.

    Args:
        data_dir: Path to the directory containing all input data files.

    Returns:
        Dictionary containing all loaded DataFrames.
    """
    print("\nLoading data files...")

    data = {}

    # Stanford RNA 3D Folding data
    stanford_dir = data_dir / "stanford-rna-3d-folding"
    if stanford_dir.exists():
        data["train_seqs"] = pd.read_csv(stanford_dir / "train_sequences.csv")
        data["valid_seqs"] = pd.read_csv(stanford_dir / "validation_sequences.csv")
        data["test_seqs"] = pd.read_csv(stanford_dir / "test_sequences.csv")
        data["train_labels"] = pd.read_csv(stanford_dir / "train_labels.csv")
        data["valid_labels"] = pd.read_csv(stanford_dir / "validation_labels.csv")
        print(
            f"Loaded Stanford data: {len(data['train_seqs'])} training, "
            f"{len(data['valid_seqs'])} validation, "
            f"{len(data['test_seqs'])} test sequences"
        )

    # Extended RNA data
    extended_dir = data_dir / "extended-rna"
    if extended_dir.exists():
        data["train_seqs_v1"] = pd.read_csv(extended_dir / "train_sequences_v2.csv")
        data["train_labels_v1"] = pd.read_csv(extended_dir / "train_labels_v2.csv")
        print(f"Loaded Extended RNA data: {len(data['train_seqs_v1'])} sequences")

    # RNA CIF to CSV data
    cif_dir = data_dir / "rna-cif-to-csv"
    if cif_dir.exists():
        data["train_seqs_v2"] = pd.read_csv(cif_dir / "rna_sequences.csv")
        data["train_labels_v2"] = pd.read_csv(cif_dir / "rna_coordinates.csv")
        print(f"Loaded CIF data: {len(data['train_seqs_v2'])} sequences")

    return data


def analyze_target_id_overlap(data: dict) -> None:
    """Analyze and print target_id relationships between datasets.

    Args:
        data: Dictionary containing loaded DataFrames.
    """
    print("\n" + "=" * 60)
    print("Dataset Analysis")
    print("=" * 60)

    datasets = []
    for key in ["train_seqs", "train_seqs_v1", "train_seqs_v2"]:
        if key in data:
            ids = set(data[key]["target_id"].unique())
            datasets.append((key, ids))
            print(f"{key} unique target_ids: {len(ids)}")

    if len(datasets) >= 2:
        all_ids = set()
        for _, ids in datasets:
            all_ids |= ids
        print(f"\nTotal unique across all datasets: {len(all_ids)}")


def merge_sequences(data: dict) -> pd.DataFrame:
    """Merge all sequence datasets into a single DataFrame.

    Args:
        data: Dictionary containing loaded DataFrames.

    Returns:
        Merged sequences DataFrame with duplicates removed.
    """
    print("\n" + "=" * 60)
    print("Merging Sequence Datasets")
    print("=" * 60)

    # Collect all sequence datasets
    seq_dfs = []
    for key in ["train_seqs", "train_seqs_v1", "train_seqs_v2"]:
        if key in data:
            seq_dfs.append(data[key])

    if not seq_dfs:
        raise ValueError("No sequence datasets found to merge")

    # Concatenate all datasets
    all_datasets = pd.concat(seq_dfs, ignore_index=True)
    print(f"Original combined rows: {len(all_datasets)}")

    # Remove duplicates based on target_id (keeping first occurrence)
    merged_train_seqs = all_datasets.drop_duplicates(subset="target_id", keep="first")
    print(f"After removing duplicates: {len(merged_train_seqs)}")
    print(f"Unique target_ids: {merged_train_seqs['target_id'].nunique()}")

    # Keep only essential columns
    merged_seqs_final = merged_train_seqs[["target_id", "sequence"]].copy()

    # Add validation sequences if available
    if "valid_seqs" in data:
        valid_seqs_subset = data["valid_seqs"][["target_id", "sequence"]].copy()
        merged_seqs_final = pd.concat(
            [merged_seqs_final, valid_seqs_subset], ignore_index=True
        )
        # Remove any duplicates that might have been introduced
        merged_seqs_final = merged_seqs_final.drop_duplicates(
            subset="target_id", keep="first"
        )
        print(f"After adding validation sequences: {len(merged_seqs_final)}")

    # Final duplicate check
    duplicates = merged_seqs_final["target_id"].duplicated().sum()
    if duplicates == 0:
        print("✓ No duplicate target_ids found - dataset is clean!")
    else:
        print(f"⚠ Warning: {duplicates} duplicate target_ids found")

    return merged_seqs_final


def merge_labels(data: dict) -> pd.DataFrame:
    """Merge all label datasets into a single DataFrame.

    Args:
        data: Dictionary containing loaded DataFrames.

    Returns:
        Merged labels DataFrame with duplicates removed.
    """
    print("\n" + "=" * 60)
    print("Merging Label Datasets")
    print("=" * 60)

    # Standard columns for labels
    label_columns = ["ID", "resname", "resid", "x_1", "y_1", "z_1"]

    # Collect all label datasets
    label_dfs = []
    for key in ["train_labels", "train_labels_v1", "train_labels_v2", "valid_labels"]:
        if key in data:
            df = data[key]
            # Ensure we only use the standard columns
            if all(col in df.columns for col in label_columns):
                label_dfs.append(df[label_columns].copy())
            else:
                print(f"⚠ Warning: {key} missing expected columns, skipping")

    if not label_dfs:
        raise ValueError("No label datasets found to merge")

    # Concatenate all label datasets
    all_labels = pd.concat(label_dfs, ignore_index=True)
    print(f"Original combined label rows: {len(all_labels)}")

    # Remove duplicates based on ID column (keeping first occurrence)
    merged_labels_final = all_labels.drop_duplicates(subset="ID", keep="first")
    print(f"After removing duplicates: {len(merged_labels_final)}")
    print(f"Unique IDs: {merged_labels_final['ID'].nunique()}")

    # Final duplicate check
    duplicates = merged_labels_final["ID"].duplicated().sum()
    if duplicates == 0:
        print("✓ No duplicate IDs found - dataset is clean!")
    else:
        print(f"⚠ Warning: {duplicates} duplicate IDs found")

    return merged_labels_final


def export_data(
    sequences_df: pd.DataFrame, labels_df: pd.DataFrame, output_dir: Path
) -> None:
    """Export merged datasets to CSV files.

    Args:
        sequences_df: Merged sequences DataFrame.
        labels_df: Merged labels DataFrame.
        output_dir: Directory to save the output files.
    """
    print("\n" + "=" * 60)
    print("Exporting Data")
    print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Export sequences
    seq_path = output_dir / "merged_sequences_final.csv"
    sequences_df.to_csv(seq_path, index=False)
    print(f"✓ Exported merged sequences to: {seq_path}")
    print(f"  Shape: {sequences_df.shape[0]:,} rows, {sequences_df.shape[1]} columns")

    # Export labels
    labels_path = output_dir / "merged_labels_final.csv"
    labels_df.to_csv(labels_path, index=False)
    print(f"✓ Exported merged labels to: {labels_path}")
    print(f"  Shape: {labels_df.shape[0]:,} rows, {labels_df.shape[1]} columns")


def main(
    data_dir: str = "data",
    output_dir: str = "data",
    sequences_only: bool = False,
    labels_only: bool = False,
) -> None:
    """Main entry point for data preparation.

    Args:
        data_dir: Path to the directory containing input data files.
                  Expected subdirectories:
                  - stanford-rna-3d-folding/
                  - extended-rna/
                  - rna-cif-to-csv/
        output_dir: Directory to save the merged output files.
        sequences_only: If True, only merge and export sequences.
        labels_only: If True, only merge and export labels.
    """
    data_path = Path(data_dir)
    output_path = Path(output_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    # Load data
    data = load_data(data_path)

    if not data:
        raise ValueError("No data files found in the specified directory")

    # Analyze target_id overlap
    analyze_target_id_overlap(data)

    # Merge and export based on flags
    if not labels_only:
        sequences_df = merge_sequences(data)
        if sequences_only:
            export_data(sequences_df, pd.DataFrame(), output_path)
            return

    if not sequences_only:
        labels_df = merge_labels(data)
        if labels_only:
            export_data(pd.DataFrame(), labels_df, output_path)
            return

    # Export both
    export_data(sequences_df, labels_df, output_path)

    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print("=" * 60)


if __name__ == "__main__":
    fire.Fire(main)
