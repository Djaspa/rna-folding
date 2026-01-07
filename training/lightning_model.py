import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleRNAFoldingModel(nn.Module):
    """
    A simple baseline model: Embedding -> LSTM -> Linear -> Coords (x, y, z)
    """

    def __init__(self, vocab_size=5, embed_dim=64, hidden_dim=128, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Linear(hidden_dim * 2, 3)  # Predict x, y, z

    def forward(self, x):
        # x: [Batch, Len]
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        coords = self.head(out)
        return coords


class RNALightningModule(pl.LightningModule):
    def __init__(
        self,
        vocab_size: int = 5,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = SimpleRNAFoldingModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        seq = batch["sequence"]
        target_coords = batch["coords"]
        mask = batch["mask"]  # Only compute loss on valid positions

        pred_coords = self(seq)

        # Loss: MSE on valid coords
        # Flatten
        pred_flat = pred_coords[mask]
        target_flat = target_coords[mask]

        loss = F.mse_loss(pred_flat, target_flat)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        seq = batch["sequence"]
        target_coords = batch["coords"]
        mask = batch["mask"]

        pred_coords = self(seq)
        pred_flat = pred_coords[mask]
        target_flat = target_coords[mask]

        loss = F.mse_loss(pred_flat, target_flat)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
