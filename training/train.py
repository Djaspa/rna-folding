import argparse
import logging
import sys
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

# Add project root to path to ensure imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dvc_utils import ensure_data_available  # noqa: E402
from training.data_module import RNADataModule  # noqa: E402
from training.lightning_model import RNALightningModule  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--fast_dev_run",
        action="store_true",
        help="Run 1 step of training/val to check bugs",
    )
    parser.add_argument(
        "--no_dvc_pull",
        action="store_true",
        help="Skip automatic DVC pull for data files",
    )
    args = parser.parse_args()

    pl.seed_everything(42)

    # Data paths
    root_dir = Path(__file__).resolve().parent.parent
    seq_csv = root_dir / "data" / "merged_sequences_final.csv"
    label_csv = root_dir / "data" / "merged_labels_final.csv"

    # Ensure data is available (auto-pull from DVC if needed)
    ensure_data_available(
        [seq_csv, label_csv],
        auto_pull=not args.no_dvc_pull,
        project_root=root_dir,
    )

    # Initialize DataModule
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
