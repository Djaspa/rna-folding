"""
Inference module for RNA folding.

Contains scripts for model registration and serving.
"""

from inference.register_model import register_model
from inference.serve import serve

__all__ = ["register_model", "serve"]
