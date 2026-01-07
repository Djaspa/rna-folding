"""
Data Processing Module

This module contains data processing utilities for the RNA folding project.
"""

from data_processing.dvc_utils import ensure_data_available

__all__ = [
    "ensure_data_available",
]


def __getattr__(name):
    """Lazy loading for modules with heavy dependencies."""
    if name == "cif_to_csv":
        from data_processing.cif_to_csv import main

        return main
    if name == "prepare_data":
        from data_processing.prepare_data import main

        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
