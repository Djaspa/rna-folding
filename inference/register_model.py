"""
Register an RNA Folding model to MLflow Model Registry.

This script loads a PyTorch Lightning checkpoint and registers it as an MLflow
model with proper input/output signatures for serving.

Usage:
    uv run python inference/register_model.py --checkpoint path/to/model.ckpt
    uv run python inference/register_model.py --checkpoint m.ckpt --model_name x
"""

import logging
import sys
from pathlib import Path

import fire
import mlflow
import mlflow.pytorch
import numpy as np
import torch
from mlflow.models import infer_signature

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.lightning_model import RNALightningModule  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class RNAModelWrapper(mlflow.pyfunc.PythonModel):
    """
    Custom MLflow pyfunc wrapper for the RNA Folding model.

    Handles preprocessing of input sequences and postprocessing of outputs.
    """

    def load_context(self, context):
        """Load the PyTorch model from artifacts."""
        import torch

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = mlflow.pytorch.load_model(
            context.artifacts["pytorch_model"], map_location=self.device
        )
        self.model.eval()

    def predict(self, context, model_input):
        """
        Predict 3D coordinates from RNA sequence indices.

        Args:
            model_input: DataFrame or ndarray with integer sequence indices.
                         Shape: (batch_size, seq_len) where values are 0-4
                         (A=0, U=1, G=2, C=3, X=4)

        Returns:
            ndarray: Predicted coordinates of shape (batch_size, seq_len, 3)
        """
        import torch

        # Convert input to tensor
        if hasattr(model_input, "values"):
            # DataFrame input
            input_data = model_input.values
        else:
            input_data = np.array(model_input)

        # Ensure 2D input
        if input_data.ndim == 1:
            input_data = input_data.reshape(1, -1)

        input_tensor = torch.tensor(input_data, dtype=torch.long, device=self.device)

        with torch.no_grad():
            output = self.model(input_tensor)

        return output.cpu().numpy()


def register_model(
    checkpoint: str,
    model_name: str = "rna-folding-model",
    tracking_uri: str = None,
    experiment: str = "rna-folding",
):
    """
    Register a trained model to MLflow.

    Args:
        checkpoint: Path to PyTorch Lightning checkpoint (.ckpt)
        model_name: Name for the registered model
        tracking_uri: MLflow tracking URI (defaults to ./plots)
        experiment: MLflow experiment name
    """
    root_dir = Path(__file__).resolve().parent.parent

    # Set tracking URI
    if tracking_uri is None:
        tracking_uri = str(root_dir / "plots")
    mlflow.set_tracking_uri(tracking_uri)
    logging.info(f"MLflow tracking URI: {tracking_uri}")

    # Set or create experiment
    mlflow.set_experiment(experiment)

    # Load the checkpoint (force CPU to avoid MPS issues)
    logging.info(f"Loading checkpoint from {checkpoint}...")
    model = RNALightningModule.load_from_checkpoint(
        checkpoint, map_location=torch.device("cpu")
    )
    model.eval()

    # Create sample input/output for signature (on CPU)
    sample_input = np.array([[0, 1, 2, 3, 1, 2, 0, 1, 2, 3]], dtype=np.int64)
    with torch.no_grad():
        sample_output = model(torch.tensor(sample_input, device="cpu")).numpy()

    signature = infer_signature(sample_input, sample_output)
    logging.info(f"Model signature: {signature}")

    # Start a run and log the model
    with mlflow.start_run(run_name=f"register-{model_name}") as run:
        logging.info(f"Started MLflow run: {run.info.run_id}")

        # Log the PyTorch model
        pytorch_model_path = "pytorch_model"
        mlflow.pytorch.log_model(model, pytorch_model_path)

        # Log the pyfunc wrapper
        artifacts = {"pytorch_model": f"runs:/{run.info.run_id}/{pytorch_model_path}"}

        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=RNAModelWrapper(),
            artifacts=artifacts,
            signature=signature,
            pip_requirements=[
                "torch>=2.0.0",
                "pytorch-lightning>=2.0.0",
                "numpy>=1.20.0",
                "mlflow>=2.0.0",
            ],
        )

        # Register the model
        model_uri = f"runs:/{run.info.run_id}/model"
        registered_model = mlflow.register_model(model_uri, model_name)
        logging.info(
            f"Model registered: {model_name} version {registered_model.version}"
        )

        return registered_model


if __name__ == "__main__":
    fire.Fire(register_model)
