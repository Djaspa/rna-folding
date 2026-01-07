import argparse
import sys
from pathlib import Path

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
    parser = argparse.ArgumentParser(description="Export RNA Folding model to ONNX")
    parser.add_argument(
        "--checkpoint_path", type=str, default=None, help="Path to .ckpt file"
    )
    parser.add_argument("--output", type=str, default="model.onnx", help="Output path")
    parser.add_argument("--vocab_size", type=int, default=5)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)

    args = parser.parse_args()

    export_model_to_onnx(
        checkpoint_path=args.checkpoint_path,
        output_path=args.output,
        vocab_size=args.vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    )
