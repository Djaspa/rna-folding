import sys
from pathlib import Path

import fire
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.lightning_model import RNALightningModule  # noqa: E402


def export_model_to_onnx(
    checkpoint_path: str = None,
    output_path: str = "model.onnx",
    vocab_size: int = 5,
    embed_dim: int = 64,
    hidden_dim: int = 128,
    num_layers: int = 2,
    model: RNALightningModule = None,
):
    """
    Export the RNA Folding model to ONNX format.

    Args:
        checkpoint_path: Path to the model checkpoint. If None and model is None,
                         initializes a fresh model (useful for testing).
        output_path: Where to save the ONNX file.
        vocab_size: Model param
        embed_dim: Model param
        hidden_dim: Model param
        num_layers: Model param
        model: Optional pre-loaded model (e.g. from training pipeline)
    """
    if model is None:
        if checkpoint_path:
            print(f"Loading model from {checkpoint_path}...")
            model = RNALightningModule.load_from_checkpoint(checkpoint_path)
        else:
            print("No checkpoint provided. Initializing fresh model for export...")
            model = RNALightningModule(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
            )

    model.eval()

    dummy_input = torch.randint(0, vocab_size, (1, 10))

    print(f"Exporting model to {output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )
    print("Export complete.")


if __name__ == "__main__":
    fire.Fire(export_model_to_onnx)
