from typing import Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from constants import PAD_IDX, VOCAB


class RNADataset(Dataset):
    def __init__(self, sequences_df, coords_df=None, max_len=None):
        self.sequences_df = sequences_df
        self.coords_df = coords_df
        self.coords_map = {}
        if self.coords_df is not None:
            # coords_df expected to have: ID (target_id_resid), x_1, y_1, z_1
            # We want to group by target_id
            # Extract target_id from ID
            self.coords_df["target_id"] = self.coords_df["ID"].apply(
                lambda x: x.rsplit("_", 1)[0]
            )
            self.coords_df["resid_idx"] = self.coords_df["ID"].apply(
                lambda x: int(x.rsplit("_", 1)[1])
            )

            for target_id, group in self.coords_df.groupby("target_id"):
                group = group.sort_values("resid_idx")
                coords = group[["x_1", "y_1", "z_1"]].values.astype(np.float32)
                self.coords_map[target_id] = coords

    def __len__(self):
        return len(self.sequences_df)

    def __getitem__(self, idx):
        row = self.sequences_df.iloc[idx]
        target_id = row["target_id"]
        sequence = row["sequence"]

        tokens = [VOCAB.get(c.upper(), 0) for c in sequence]
        tokens_tensor = torch.tensor(tokens, dtype=torch.long)

        coords_tensor = torch.zeros((len(sequence), 3), dtype=torch.float32)
        mask = torch.zeros(len(sequence), dtype=torch.bool)

        if target_id in self.coords_map:
            coords = self.coords_map[target_id]
            L = len(sequence)
            C = len(coords)
            valid_len = min(L, C)

            coords_tensor[:valid_len] = torch.tensor(coords[:valid_len])
            mask[:valid_len] = True

        return {
            "target_id": target_id,
            "sequence": tokens_tensor,
            "coords": coords_tensor,
            "mask": mask,
            "len": len(sequence),
        }


def collate_fn(batch):
    # Padding
    max_len = max(item["len"] for item in batch)

    padded_seqs = []
    padded_coords = []
    padded_masks = []

    for item in batch:
        L = item["len"]
        seq = item["sequence"]
        coords = item["coords"]
        mask = item["mask"]

        # Pad sequence
        seq_pad = torch.nn.functional.pad(seq, (0, max_len - L), value=PAD_IDX)
        padded_seqs.append(seq_pad)

        # Pad coords
        coords_pad = torch.nn.functional.pad(coords, (0, 0, 0, max_len - L), value=0.0)
        padded_coords.append(coords_pad)

        # Pad mask
        mask_pad = torch.nn.functional.pad(mask, (0, max_len - L), value=False)
        padded_masks.append(mask_pad)

    return {
        "sequence": torch.stack(padded_seqs),
        "coords": torch.stack(padded_coords),
        "mask": torch.stack(padded_masks),
    }


class RNADataModule(pl.LightningDataModule):
    def __init__(
        self,
        seq_csv: str = "merged_sequences_final.csv",
        label_csv: str = "merged_labels_final.csv",
        batch_size: int = 4,
        num_workers: int = 0,
        val_split: float = 0.1,
    ):
        super().__init__()
        self.seq_csv = seq_csv
        self.label_csv = label_csv
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split

    def setup(self, stage: Optional[str] = None):
        # Load data
        seq_df = pd.read_csv(self.seq_csv)
        label_df = pd.read_csv(self.label_csv)

        # Filter sequences that have labels
        # Identify target_ids present in labels
        labeled_ids = set(label_df["ID"].apply(lambda x: x.rsplit("_", 1)[0]).unique())

        # Separate labeled and unlabeled if needed. For now, use intersection
        train_df = seq_df[seq_df["target_id"].isin(labeled_ids)].reset_index(drop=True)

        dataset = RNADataset(train_df, label_df)

        val_size = int(len(dataset) * self.val_split)
        train_size = len(dataset) - val_size

        self.train_ds, self.val_ds = random_split(
            dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
        )
