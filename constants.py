"""Global constants for RNA folding project.

These are fixed values that are not meant to be configurable
and represent fundamental properties of RNA sequences and structures.
"""

# RNA vocabulary mapping nucleotides to indices
# T is treated as U (Thymine -> Uracil for RNA)
VOCAB = {"A": 1, "C": 2, "G": 3, "U": 4, "T": 4}

# Padding index for sequence padding
PAD_IDX = 0

# Unknown nucleotide index (maps to padding)
UNK_IDX = 0

# Complementary base pairs for RNA
COMPLEMENTARY_BASES = {"A": "U", "U": "A", "G": "C", "C": "G"}

# RNA structural constants (Angstroms)
RNA_BACKBONE_DISTANCE = 5.9  # Typical C1'-C1' distance
MIN_STEM_LENGTH = 3  # Minimum length for stem detection
