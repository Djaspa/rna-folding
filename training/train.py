import argparse
import sys
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from training.data_module import RNADataModule
from training.lightning_model import RNALightningModule

# Add project root to path to ensure imports work
sys.path.append(str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--fast_dev_run",
        action="store_true",
        help="Run 1 step of training/val to check bugs",
    )
    args = parser.parse_args()

    pl.seed_everything(42)

    # Initialize DataModule
    root_dir = Path(__file__).resolve().parent.parent
    seq_csv = root_dir / "merged_sequences_final.csv"
    label_csv = root_dir / "merged_labels_final.csv"

    dm = RNADataModule(seq_csv=seq_csv, label_csv=label_csv, batch_size=args.batch_size)

    # Initialize Model
    model = RNALightningModule()

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=root_dir / "checkpoints",
        filename="rna-fold-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        fast_dev_run=args.fast_dev_run,
        callbacks=[checkpoint_callback],
        accelerator="auto",
        devices="auto",
    )

    # Train
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
