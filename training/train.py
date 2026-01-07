import logging
import sys
from pathlib import Path

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

# Add project root to path to ensure imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dvc_utils import ensure_data_available  # noqa: E402
from training.data_module import RNADataModule  # noqa: E402
from training.lightning_model import RNALightningModule  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed)

    # Data paths
    root_dir = Path(__file__).resolve().parent.parent
    seq_csv = root_dir / cfg.paths.data_dir / cfg.data.seq_csv
    label_csv = root_dir / cfg.paths.data_dir / cfg.data.label_csv

    # Ensure data is available (auto-pull from DVC if needed)
    ensure_data_available(
        [seq_csv, label_csv],
        auto_pull=not cfg.training.no_dvc_pull,
        project_root=root_dir,
    )

    # Initialize DataModule
    dm = RNADataModule(
        seq_csv=seq_csv,
        label_csv=label_csv,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
        val_split=cfg.data.val_split,
    )

    # Initialize Model
    model = RNALightningModule(
        vocab_size=cfg.model.vocab_size,
        embed_dim=cfg.model.embed_dim,
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers,
        lr=cfg.model.learning_rate,
    )

    # Initialize Logger
    logger = None
    if cfg.logging.enabled:
        tracking_uri = cfg.logging.tracking_uri
        # If tracking_uri is a relative path (like "mlruns"), make it absolute
        if not tracking_uri.startswith("http") and not Path(tracking_uri).is_absolute():
            tracking_uri = str(root_dir / tracking_uri)

        logger = MLFlowLogger(
            experiment_name=cfg.logging.experiment_name,
            tracking_uri=tracking_uri,
            run_name=cfg.logging.run_name,
            log_model=cfg.logging.log_model,
            tags=cfg.logging.tags,
        )

        # Log all hyperparameters
        logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
        logging.info(f"MLflow logging enabled. Tracking URI: {tracking_uri}")
    else:
        logging.info("MLflow logging disabled.")

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=root_dir / cfg.paths.checkpoints_dir,
        filename=cfg.training.checkpoint.filename,
        save_top_k=cfg.training.checkpoint.save_top_k,
        monitor=cfg.training.checkpoint.monitor,
        mode=cfg.training.checkpoint.mode,
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        fast_dev_run=cfg.training.fast_dev_run,
        callbacks=[checkpoint_callback],
        logger=logger,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
    )

    # Train
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
