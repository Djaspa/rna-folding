"""
Start MLflow Model Serving endpoint.

This script starts an MLflow model serving endpoint for the RNA Folding model.

Usage:
    uv run python scripts/serve.py
    uv run python scripts/serve.py --port 8080
    uv run python scripts/serve.py --model_version 1
"""

import logging
import os
import subprocess
import sys
from pathlib import Path

import fire
import yaml

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_serving_config(config_path: str = None) -> dict:
    """Load serving configuration from YAML file."""
    if config_path is None:
        root_dir = Path(__file__).resolve().parent.parent
        config_path = root_dir / "configs" / "serving" / "mlflow_serving.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def get_model_uri(model_name: str, model_version: str, tracking_uri: str) -> str:
    """
    Get the model URI for serving.

    Args:
        model_name: Registered model name
        model_version: Version number or "latest"
        tracking_uri: MLflow tracking URI

    Returns:
        Model URI string
    """
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)

    if model_version == "latest":
        # Get the latest version
        client = mlflow.tracking.MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            raise ValueError(f"No versions found for model '{model_name}'")
        # Sort by version number and get the latest
        latest = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
        model_version = latest.version
        logging.info(f"Using latest version: {model_version}")

    return f"models:/{model_name}/{model_version}"


def serve(
    host: str = None,
    port: int = None,
    model_name: str = None,
    model_version: str = None,
    workers: int = None,
    tracking_uri: str = None,
    config: str = None,
):
    """
    Start the MLflow model serving endpoint.

    Args:
        host: Host address to bind to (default: 127.0.0.1)
        port: Port number (default: 5001)
        model_name: Registered model name (default: rna-folding-model)
        model_version: Model version or "latest" (default: latest)
        workers: Number of worker processes (default: 1)
        tracking_uri: MLflow tracking URI (default: ./plots)
        config: Path to serving config YAML
    """
    root_dir = Path(__file__).resolve().parent.parent

    # Load config from file
    cfg = load_serving_config(config)

    # Apply defaults from config, then override with CLI args
    host = host or cfg.get("host", "127.0.0.1")
    port = port or cfg.get("port", 5001)
    model_name = model_name or cfg.get("model_name", "rna-folding-model")
    model_version = model_version or cfg.get("model_version", "latest")
    workers = workers or cfg.get("workers", 1)

    if tracking_uri is None:
        tracking_uri = str(root_dir / "plots")

    # Get the model URI
    try:
        model_uri = get_model_uri(model_name, str(model_version), tracking_uri)
        logging.info(f"Serving model: {model_uri}")
    except ValueError as e:
        logging.error(str(e))
        logging.error(
            "Please register a model first using: "
            "uv run python scripts/register_model.py --checkpoint <path>"
        )
        sys.exit(1)

    # Build the mlflow serve command
    cmd = [
        sys.executable,
        "-m",
        "mlflow",
        "models",
        "serve",
        "-m",
        model_uri,
        "--host",
        host,
        "--port",
        str(port),
        "--workers",
        str(workers),
        "--env-manager",
        "local",  # Use local environment instead of creating new conda env
    ]

    # Set the tracking URI in environment
    env = {"MLFLOW_TRACKING_URI": tracking_uri}

    logging.info(f"Starting MLflow serving on http://{host}:{port}")
    logging.info(f"Model: {model_name} (version {model_version})")
    logging.info("Press Ctrl+C to stop")
    logging.info("-" * 50)

    try:
        # Run the serve command, inheriting environment
        full_env = os.environ.copy()
        full_env.update(env)

        process = subprocess.Popen(cmd, env=full_env)
        process.wait()
    except KeyboardInterrupt:
        logging.info("\nShutting down server...")
        process.terminate()


if __name__ == "__main__":
    fire.Fire(serve)
