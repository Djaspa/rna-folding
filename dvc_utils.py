"""DVC utility functions for data management."""

import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def ensure_data_available(
    data_paths: list[Path],
    auto_pull: bool = True,
    project_root: Path | None = None,
) -> None:
    """Ensure required data files are available, optionally pulling from DVC.

    Args:
        data_paths: List of paths to required data files.
        auto_pull: If True, run `dvc pull` when files are missing.
        project_root: Project root directory for running dvc commands.
                      Defaults to parent of this file.

    Raises:
        FileNotFoundError: If files are missing and auto_pull is False
                           or dvc pull fails.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent

    missing_files = [p for p in data_paths if not p.exists()]

    if not missing_files:
        logger.debug("All required data files are present.")
        return

    if not auto_pull:
        missing_str = "\n  - ".join(str(p) for p in missing_files)
        raise FileNotFoundError(
            f"Required data files are missing:\n  - {missing_str}\n"
            "Run 'dvc pull' to download the data, or remove --no-dvc-pull flag."
        )

    logger.info("Some data files are missing. Running 'dvc pull'...")
    logger.info("Missing files: %s", [p.name for p in missing_files])

    try:
        result = subprocess.run(
            [sys.executable, "-m", "dvc", "pull"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info("DVC pull completed successfully.")
        if result.stdout:
            logger.debug("DVC output: %s", result.stdout)
    except subprocess.CalledProcessError as e:
        logger.error("DVC pull failed: %s", e.stderr)
        raise FileNotFoundError(
            f"Failed to download data via DVC. Error:\n{e.stderr}\n"
            "Please run 'dvc pull' manually and check your DVC configuration."
        ) from e

    # Verify files are now present
    still_missing = [p for p in data_paths if not p.exists()]
    if still_missing:
        missing_str = "\n  - ".join(str(p) for p in still_missing)
        raise FileNotFoundError(
            f"DVC pull completed but files are still missing:\n  - {missing_str}\n"
            "Check that these files are tracked by DVC."
        )

    logger.info("All required data files are now available.")
