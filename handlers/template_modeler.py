import random

import numpy as np
from Bio import pairwise2
from Bio.Seq import Seq
from scipy.spatial import distance_matrix
from scipy.spatial.transform import Rotation as R


# Vectorized version of process_labels function
def process_labels_vectorized(labels_df):
    # Extract target_id from ID column (remove last part after underscore)
    labels_df = labels_df.copy()
    labels_df["target_id"] = labels_df["ID"].str.rsplit("_", n=1).str[0]

    # Sort by target_id and resid for proper ordering
    labels_df = labels_df.sort_values(["target_id", "resid"])

    # Group by target_id and convert coordinates to arrays
    coords_dict = {}
    for target_id, group in labels_df.groupby("target_id"):
        # Extract coordinates as numpy array in one operation
        coords_dict[target_id] = group[["x_1", "y_1", "z_1"]].values

    return coords_dict


def find_similar_sequences(query_seq, train_seqs_df, train_coords_dict, top_n=5):
    similar_seqs = []
    query_seq_obj = Seq(query_seq)

    for _, row in train_seqs_df.iterrows():
        target_id = row["target_id"]
        train_seq = row["sequence"]

        # Skip if coordinates not available
        if target_id not in train_coords_dict:
            continue

        # Skip if sequence is too different in length (more than 40% difference)
        if (
            abs(len(train_seq) - len(query_seq)) / max(len(train_seq), len(query_seq))
            > 0.4
        ):
            continue

        # Perform sequence alignment
        alignments = pairwise2.align.globalms(
            query_seq_obj, train_seq, 2.9, -1, -10, -0.5, one_alignment_only=True
        )

        if alignments:
            alignment = alignments[0]
            similarity_score = alignment.score / (
                2 * min(len(query_seq), len(train_seq))
            )
            similar_seqs.append(
                (target_id, train_seq, similarity_score, train_coords_dict[target_id])
            )

    # Sort by similarity score (higher is better) and return top N
    similar_seqs.sort(key=lambda x: x[2], reverse=True)
    return similar_seqs[:top_n]


# ======= adaptive_rna_constraints =================
def adaptive_rna_constraints(coordinates, sequence, confidence=1.0):
    """Apply realistic RNA structural constraints"""
    # Make a copy of coordinates to refine
    refined_coords = coordinates.copy()
    n_residues = len(sequence)

    # Calculate constraint strength (inverse of confidence)
    constraint_strength = 0.8 * (1.0 - min(confidence, 0.8))

    # 1. Sequential distance constraints (consecutive nucleotides)
    seq_min_dist = 5.5  # Minimum sequential distance
    seq_max_dist = 6.5  # Maximum sequential distance

    for i in range(n_residues - 1):
        current_pos = refined_coords[i]
        next_pos = refined_coords[i + 1]

        # Calculate current distance
        current_dist = np.linalg.norm(next_pos - current_pos)

        # Only adjust if significantly outside expected range
        if current_dist < seq_min_dist or current_dist > seq_max_dist:
            # Calculate target distance (midpoint of range)
            target_dist = (seq_min_dist + seq_max_dist) / 2

            # Get direction vector
            direction = next_pos - current_pos
            direction = direction / (np.linalg.norm(direction) + 1e-10)

            # Apply partial adjustment based on constraint strength
            adjustment = (target_dist - current_dist) * constraint_strength

            # Only adjust the next position to preserve the overall fold
            refined_coords[i + 1] = current_pos + direction * (
                current_dist + adjustment
            )

    # 2. Steric clash prevention
    min_allowed_distance = 3.8  # Minimum distance between non-consecutive C1' atoms

    # Calculate all pairwise distances
    dist_matrix = distance_matrix(refined_coords, refined_coords)

    # Find severe clashes (atoms too close)
    severe_clashes = np.where((dist_matrix < min_allowed_distance) & (dist_matrix > 0))

    # Fix severe clashes
    for idx in range(len(severe_clashes[0])):
        i, j = severe_clashes[0][idx], severe_clashes[1][idx]

        # Skip consecutive nucleotides and previously processed pairs
        if abs(i - j) <= 1 or i >= j:
            continue

        # Get current positions and distance
        pos_i = refined_coords[i]
        pos_j = refined_coords[j]
        current_dist = dist_matrix[i, j]

        # Calculate necessary adjustment but scale by constraint strength
        direction = pos_j - pos_i
        direction = direction / (np.linalg.norm(direction) + 1e-10)

        # Calculate partial adjustment
        adjustment = (min_allowed_distance - current_dist) * constraint_strength

        # Move points apart
        refined_coords[i] = pos_i - direction * (adjustment / 2)
        refined_coords[j] = pos_j + direction * (adjustment / 2)

    return refined_coords


def adapt_template_to_query(query_seq, template_seq, template_coords, alignment=None):
    if alignment is None:
        from Bio import pairwise2
        from Bio.Seq import Seq

        query_seq_obj = Seq(query_seq)
        template_seq_obj = Seq(template_seq)
        alignments = pairwise2.align.globalms(
            query_seq_obj, template_seq_obj, 2.9, -1, -10, -0.5, one_alignment_only=True
        )

        if not alignments:
            return generate_improved_rna_structure(query_seq)

        alignment = alignments[0]

    aligned_query = alignment.seqA
    aligned_template = alignment.seqB

    query_coords = np.zeros((len(query_seq), 3))
    query_coords.fill(np.nan)

    # Map template coordinates to query
    query_idx = 0
    template_idx = 0

    for i in range(len(aligned_query)):
        query_char = aligned_query[i]
        template_char = aligned_template[i]

        if query_char != "-" and template_char != "-":
            if template_idx < len(template_coords):
                query_coords[query_idx] = template_coords[template_idx]
            template_idx += 1
            query_idx += 1
        elif query_char != "-" and template_char == "-":
            query_idx += 1
        elif query_char == "-" and template_char != "-":
            template_idx += 1

    # IMPROVED GAP FILLING - maintains RNA backbone geometry
    backbone_distance = 5.9  # Typical C1'-C1' distance

    # Fill gaps by maintaining realistic backbone connectivity
    for i in range(len(query_coords)):
        if np.isnan(query_coords[i, 0]):
            # Find nearest valid neighbors
            prev_valid = next_valid = None

            for j in range(i - 1, -1, -1):
                if not np.isnan(query_coords[j, 0]):
                    prev_valid = j
                    break

            for j in range(i + 1, len(query_coords)):
                if not np.isnan(query_coords[j, 0]):
                    next_valid = j
                    break

            if prev_valid is not None and next_valid is not None:
                # Interpolate along realistic RNA backbone path
                gap_size = next_valid - prev_valid
                total_distance = np.linalg.norm(
                    query_coords[next_valid] - query_coords[prev_valid]
                )
                expected_distance = gap_size * backbone_distance

                # If gap is compressed, extend it realistically
                if total_distance < expected_distance * 0.7:
                    direction = query_coords[next_valid] - query_coords[prev_valid]
                    direction = direction / (np.linalg.norm(direction) + 1e-10)

                    # Place intermediate points along extended path
                    for k, idx in enumerate(range(prev_valid + 1, next_valid)):
                        progress = (k + 1) / gap_size
                        base_pos = (
                            query_coords[prev_valid]
                            + direction * expected_distance * progress
                        )

                        # Add slight curvature for realism
                        perpendicular = np.cross(direction, [0, 0, 1])
                        if np.linalg.norm(perpendicular) < 1e-6:
                            perpendicular = np.cross(direction, [1, 0, 0])
                        perpendicular = perpendicular / (
                            np.linalg.norm(perpendicular) + 1e-10
                        )

                        curve_amplitude = 2.0 * np.sin(progress * np.pi)
                        query_coords[idx] = base_pos + perpendicular * curve_amplitude
                else:
                    # Linear interpolation for normal gaps
                    for k, idx in enumerate(range(prev_valid + 1, next_valid)):
                        weight = (k + 1) / gap_size
                        query_coords[idx] = (1 - weight) * query_coords[
                            prev_valid
                        ] + weight * query_coords[next_valid]

            elif prev_valid is not None:
                # Extend from previous position
                if prev_valid > 0 and not np.isnan(query_coords[prev_valid - 1, 0]):
                    direction = query_coords[prev_valid] - query_coords[prev_valid - 1]
                    direction = direction / (np.linalg.norm(direction) + 1e-10)
                else:
                    direction = np.array([1.0, 0.0, 0.0])

                steps_needed = i - prev_valid
                for step in range(1, steps_needed + 1):
                    pos_idx = prev_valid + step
                    if pos_idx < len(query_coords):
                        query_coords[pos_idx] = (
                            query_coords[prev_valid]
                            + direction * backbone_distance * step
                        )

            elif next_valid is not None:
                # Work backwards from next position
                direction = np.array([-1.0, 0.0, 0.0])  # Default backward direction
                steps_needed = next_valid - i
                for step in range(steps_needed, 0, -1):
                    pos_idx = next_valid - step
                    if pos_idx >= 0:
                        query_coords[pos_idx] = (
                            query_coords[next_valid]
                            - direction * backbone_distance * step
                        )

    # Final cleanup
    query_coords = np.nan_to_num(query_coords)
    return query_coords


# ========== generate_improved_rna_structure ========================
def generate_improved_rna_structure(sequence):
    """
    Generate a more realistic RNA structure fallback based on sequence patterns
    and basic RNA structure principles.

    Args:
        sequence: RNA sequence string

    Returns:
        Array of 3D coordinates
    """
    n_residues = len(sequence)
    coordinates = np.zeros((n_residues, 3))

    # Analyze sequence to predict structural elements
    # Look for complementary regions that could form base pairs
    potential_stems = identify_potential_stems(sequence)

    # Default parameters
    radius_helix = 10.0
    radius_loop = 15.0
    rise_per_residue_helix = 2.5
    rise_per_residue_loop = 1.5
    angle_per_residue_helix = 0.6
    angle_per_residue_loop = 0.3

    # Assign structural classifications
    structure_types = assign_structure_types(sequence, potential_stems)

    # Generate coordinates based on predicted structure
    current_pos = np.array([0.0, 0.0, 0.0])
    # current_direction = np.array([0.0, 0.0, 1.0])
    current_angle = 0.0

    for i in range(n_residues):
        if structure_types[i] == "stem":
            # Part of a helical stem
            current_angle += angle_per_residue_helix
            coordinates[i] = [
                radius_helix * np.cos(current_angle),
                radius_helix * np.sin(current_angle),
                current_pos[2] + rise_per_residue_helix,
            ]
            current_pos = coordinates[i]
        elif structure_types[i] == "loop":
            # Part of a loop
            current_angle += angle_per_residue_loop
            z_shift = rise_per_residue_loop * np.sin(current_angle * 0.5)
            coordinates[i] = [
                radius_loop * np.cos(current_angle),
                radius_loop * np.sin(current_angle),
                current_pos[2] + z_shift,
            ]
            current_pos = coordinates[i]
        else:
            # Single-stranded region
            # Add some randomness to make it look more realistic
            jitter = np.random.normal(0, 1, 3) * 2.0
            coordinates[i] = current_pos + jitter
            current_pos = coordinates[i]

    return coordinates


def identify_potential_stems(sequence):
    """
    Identify potential stem regions by looking for self-complementary segments.

    Args:
        sequence: RNA sequence string

    Returns:
        List of tuples (start1, end1, start2, end2) representing
        potentially paired regions
    """
    complementary_bases = {"A": "U", "U": "A", "G": "C", "C": "G"}
    min_stem_length = 3
    potential_stems = []

    # Simple stem identification
    for i in range(len(sequence) - min_stem_length):
        for j in range(i + min_stem_length + 3, len(sequence) - min_stem_length + 1):
            # Check if regions could form a stem
            potential_stem_len = min(min_stem_length, len(sequence) - j)
            is_stem = True

            for k in range(potential_stem_len):
                if (
                    sequence[i + k] not in complementary_bases
                    or complementary_bases[sequence[i + k]]
                    != sequence[j + potential_stem_len - k - 1]
                ):
                    is_stem = False
                    break

            if is_stem:
                potential_stems.append(
                    (i, i + potential_stem_len - 1, j, j + potential_stem_len - 1)
                )

    return potential_stems


def assign_structure_types(sequence, potential_stems):
    """
    Assign each nucleotide to a structural element type.

    Args:
        sequence: RNA sequence string
        potential_stems: List of tuples representing stem regions

    Returns:
        List of structure types ('stem', 'loop', 'single')
    """
    structure_types = ["single"] * len(sequence)

    # Mark stem regions
    for stem in potential_stems:
        start1, end1, start2, end2 = stem
        for i in range(end1 - start1 + 1):
            structure_types[start1 + i] = "stem"
            structure_types[end2 - i] = "stem"

    # Mark loop regions (regions between paired regions)
    for i in range(len(potential_stems) - 1):
        _, end1, start2, _ = potential_stems[i]
        next_start1, _, _, _ = potential_stems[i + 1]

        if next_start1 > end1 + 1 and start2 > next_start1:
            for j in range(end1 + 1, next_start1):
                structure_types[j] = "loop"

    return structure_types


# =========== generate_rna_structure ======================
def generate_rna_structure(sequence, seed=None):
    """Generate a more realistic RNA structure when no good templates are found"""
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    n_residues = len(sequence)
    coordinates = np.zeros((n_residues, 3))

    # Initialize the first few residues in a helix
    for i in range(min(3, n_residues)):
        angle = i * 0.6
        coordinates[i] = [10.0 * np.cos(angle), 10.0 * np.sin(angle), i * 2.5]

    # Add more complex folding patterns
    current_direction = np.array([0.0, 0.0, 1.0])  # Start moving along z-axis

    # Define base-pairing tendencies (G-C and A-U pairs)
    for i in range(3, n_residues):
        # Check for potential base-pairing in the sequence
        has_pair = False
        pair_idx = -1

        # Simple detection of complementary bases (G-C, A-U)
        complementary = {"G": "C", "C": "G", "A": "U", "U": "A"}
        current_base = sequence[i]

        # Look for potential base-pairing within a window before the current position
        window_size = min(i, 15)  # Look back up to 15 bases
        for j in range(i - window_size, i):
            if j >= 0 and sequence[j] == complementary.get(current_base, "X"):
                # Found a potential pair
                has_pair = True
                pair_idx = j
                break

        if has_pair and i - pair_idx <= 10 and random.random() < 0.7:
            # Try to create a base-pair by positioning this nucleotide near its pair
            pair_pos = coordinates[pair_idx]

            # Create a position that's roughly opposite to the pair
            random_offset = np.random.normal(0, 1, 3) * 2.0
            base_pair_distance = 10.0 + random.uniform(-1.0, 1.0)

            # Calculate a vector from base-pair toward center of structure
            center = np.mean(coordinates[:i], axis=0)
            direction = center - pair_pos
            direction = direction / (np.linalg.norm(direction) + 1e-10)

            # Position new nucleotide in the general direction of the "center"
            coordinates[i] = pair_pos + direction * base_pair_distance + random_offset

            # Update direction for next nucleotide
            current_direction = np.random.normal(0, 0.3, 3)
            current_direction = current_direction / (
                np.linalg.norm(current_direction) + 1e-10
            )

        else:
            # No base-pairing detected, continue with the current fold direction
            # Randomly rotate current direction to simulate RNA flexibility
            if random.random() < 0.3:
                # More significant direction change
                angle = random.uniform(0.2, 0.6)
                axis = np.random.normal(0, 1, 3)
                axis = axis / (np.linalg.norm(axis) + 1e-10)
                rotation = R.from_rotvec(angle * axis)
                current_direction = rotation.apply(current_direction)
            else:
                # Small random changes in direction
                current_direction += np.random.normal(0, 0.15, 3)
                current_direction = current_direction / (
                    np.linalg.norm(current_direction) + 1e-10
                )

            # Distance between consecutive nucleotides (3.5-4.5Å is typical)
            step_size = random.uniform(3.5, 4.5)

            # Update position
            coordinates[i] = coordinates[i - 1] + step_size * current_direction

    return coordinates


# ========== predict_rna_structures ==================
def predict_rna_structures(
    sequence, target_id, train_seqs_df, train_coords_dict, n_predictions=5
):
    predictions = []

    # Find similar sequences in the training data
    similar_seqs = find_similar_sequences(
        sequence, train_seqs_df, train_coords_dict, top_n=n_predictions
    )

    # If we found any similar sequences, use them as templates
    if similar_seqs:
        for i, (
            template_id,
            template_seq,
            similarity_score,
            template_coords,
        ) in enumerate(similar_seqs):
            # Adapt template coordinates to the query sequence
            adapted_coords = adapt_template_to_query(
                sequence, template_seq, template_coords
            )

            if adapted_coords is not None:
                # Apply adaptive constraints based on template similarity
                # For high similarity templates, apply very gentle constraints
                refined_coords = adaptive_rna_constraints(
                    adapted_coords, sequence, confidence=similarity_score
                )

                # Add some randomness (less for better templates)
                random_scale = max(0.05, 0.8 - similarity_score)  # Reduced randomness
                randomized_coords = refined_coords.copy()
                randomized_coords += np.random.normal(
                    0, random_scale, randomized_coords.shape
                )

                predictions.append(randomized_coords)

                if len(predictions) >= n_predictions:
                    break

    # If we don't have enough predictions from templates, generate de novo structures
    while len(predictions) < n_predictions:
        seed_value = hash(target_id) % 10000 + len(predictions) * 1000
        de_novo_coords = generate_rna_structure(sequence, seed=seed_value)

        # Apply stronger constraints to de novo structures (lower confidence)
        refined_de_novo = adaptive_rna_constraints(
            de_novo_coords, sequence, confidence=0.2
        )

        predictions.append(refined_de_novo)

    return predictions[:n_predictions]
