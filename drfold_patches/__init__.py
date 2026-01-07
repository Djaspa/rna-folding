"""
DRfold Patches Module

This module contains patched versions of DRfold2 components
for RNA 3D structure prediction.
"""

from drfold_patches.cubic import (
    get_cubic_spline_coefs,
    point_to_point_cubic_hermite,
)
from drfold_patches.operations import (
    init_coordinates,
    norm_atom_locations,
    norm_features,
    read_fasta_file,
)
from drfold_patches.optimization import (
    Optimization,
)
from drfold_patches.selection import (
    Selection,
)

__all__ = [
    "get_cubic_spline_coefs",
    "point_to_point_cubic_hermite",
    "init_coordinates",
    "norm_atom_locations",
    "norm_features",
    "read_fasta_file",
    "Optimization",
    "Selection",
]
