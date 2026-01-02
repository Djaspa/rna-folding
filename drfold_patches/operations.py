"""
operations.py: Core Mathematical Operations for RNA Structure Analysis

This module provides essential mathematical operations for manipulating and analyzing
RNA 3D structures, organized into four main categories:

1. Basic Vector Operations:
   Functions for selecting coordinates and calculating distances between points,
   which form the foundation for all structural calculations.

2. Angle Calculations:
   Functions for computing bond angles and dihedral (torsion) angles between atoms,
   with differentiable implementations suitable for gradient-based optimization.

3. Rigid Body Transformations:
   Functions for determining optimal rotations and translations between sets of
   coordinates, enabling structure alignment and manipulation.

4. Sequence Utilities:
   Functions for converting RNA sequence data into standard 3D coordinate templates,
   allowing sequence-structure mapping.

These operations support the core functionality of RNA structure prediction, analysis,
and optimization throughout the codebase.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Function

# Use consistent epsilon value across all functions
EPS = 1e-8


# === Basic Vector Operations ===
def coor_selection(coor, mask):
    # [L,n,3],[L,n],byte
    return torch.masked_select(coor, mask.bool()).view(-1, 3)


def pair_distance(x1, x2, eps=1e-6, p=2):
    # Use torch.cdist for p=2 (Euclidean) which is highly optimized
    if p == 2:
        return torch.cdist(x1, x2, p=2)

    # For other p-norms, avoid memory expansion with broadcasting
    x1_ = x1.unsqueeze(1)  # [n1, 1, dim]
    x2_ = x2.unsqueeze(0)  # [1, n2, dim]
    diff = torch.abs(x1_ - x2_)
    out = torch.pow(diff + eps, p).sum(dim=2)
    return torch.pow(out, 1.0 / p)


# === Angle Calculations ===
def angle(p0, p1, p2):
    # [b 3]
    b0 = p0 - p1
    b1 = p2 - p1

    b0 = b0 / (torch.norm(b0, dim=-1, keepdim=True) + EPS)
    b1 = b1 / (torch.norm(b1, dim=-1, keepdim=True) + EPS)

    recos = torch.sum(b0 * b1, -1)
    recos = torch.clamp(recos, -0.9999, 0.9999)
    return torch.acos(recos)


class torsion(Function):
    # PyTorch class to calculate differentiable torsion angle
    # https://stackoverflow.com/questions/20305272/dihedral-torsion-angle-from-four-points-in-cartesian-coordinates-in-python
    # https://salilab.org/modeller/manual/node492.html
    @staticmethod
    def forward(ctx, p0, p1, p2, p3):
        # Save input points for backward pass
        ctx.save_for_backward(p0, p1, p2, p3)

        # Calculate bond vectors
        b0 = p0 - p1
        b1 = p2 - p1
        b2 = p3 - p2

        # Normalize the middle bond vector
        b1_norm = torch.norm(b1, dim=-1, keepdim=True) + 1e-8
        b1_unit = b1 / b1_norm

        # Project the other bonds onto the plane perpendicular to middle bond
        v = b0 - torch.sum(b0 * b1_unit, dim=-1, keepdim=True) * b1_unit
        w = b2 - torch.sum(b2 * b1_unit, dim=-1, keepdim=True) * b1_unit

        # Calculate torsion using the arctan2 formula (more stable than arccos)
        x = torch.sum(v * w, dim=-1)  # cosine component
        y = torch.sum(torch.cross(b1_unit, v, dim=-1) * w, dim=-1)  # sine component

        return torch.atan2(y, x)

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors from forward pass
        p0, p1, p2, p3 = ctx.saved_tensors

        # Calculate bond vectors
        r01 = p0 - p1
        r12 = p2 - p1
        r23 = p3 - p2

        # Calculate bond lengths with numerical stability
        d01 = torch.norm(r01, dim=-1, keepdim=True) + 1e-8
        d12 = torch.norm(r12, dim=-1, keepdim=True) + 1e-8
        d23 = torch.norm(r23, dim=-1, keepdim=True) + 1e-8

        # Normalize bond vectors
        e01 = r01 / d01
        e12 = r12 / d12
        e23 = r23 / d23

        # Calculate normal vectors to the two planes
        n1 = torch.cross(e01, e12, dim=-1)
        n2 = torch.cross(e12, e23, dim=-1)

        # Normalize normal vectors
        n1_norm = torch.norm(n1, dim=-1, keepdim=True) + 1e-8
        n2_norm = torch.norm(n2, dim=-1, keepdim=True) + 1e-8
        n1 = n1 / n1_norm
        n2 = n2 / n2_norm

        # Calculate gradients for each atom
        # These are based on the analytical derivatives of dihedral angles
        g0 = torch.cross(e01, n1, dim=-1) / d01
        g1 = -g0 - torch.cross(e12, n1, dim=-1) / d12
        g2 = torch.cross(e12, n2, dim=-1) / d12 - torch.cross(e23, n2, dim=-1) / d23
        g3 = torch.cross(e23, n2, dim=-1) / d23

        # Apply chain rule with incoming gradient
        g0 = g0 * grad_output.unsqueeze(-1)
        g1 = g1 * grad_output.unsqueeze(-1)
        g2 = g2 * grad_output.unsqueeze(-1)
        g3 = g3 * grad_output.unsqueeze(-1)

        return g0, g1, g2, g3


def dihedral(input1, input2, input3, input4):
    return torsion.apply(input1, input2, input3, input4)


# === Rigid Body Transformations ===
def rigidFrom3Points(x):
    x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
    v1 = x3 - x2
    v2 = x1 - x2

    # Normalize v1 to get e1
    e1 = F.normalize(v1, p=2, dim=-1)

    # Project v2 onto e1 and subtract to get the component orthogonal to e1
    u2 = v2 - e1 * (torch.einsum("bn,bn->b", e1, v2)[:, None])

    # Normalize u2 to get e2
    e2 = F.normalize(u2, p=2, dim=-1)

    # Cross product to get e3
    e3 = torch.cross(e1, e2, dim=-1)

    return torch.stack([e1, e2, e3], dim=1)


# return the direction from to_q to from_p
def Kabsch_rigid(bases, x1, x2, x3):
    # Early return for empty input
    if x1.shape[0] == 0:
        return torch.empty(0, 3, 3), torch.empty(0, 3)

    the_dim = 1
    to_q = torch.stack([x1, x2, x3], dim=the_dim)
    biasq = torch.mean(to_q, dim=the_dim, keepdim=True)
    q = to_q - biasq
    m = torch.einsum("bnz,bny->bzy", bases, q)
    u, s, v = torch.svd(m)
    vt = torch.transpose(v, 1, 2)
    det = torch.det(torch.matmul(u, vt))
    det = det.view(-1, 1, 1)
    vt = torch.cat((vt[:, :2, :], vt[:, -1:, :] * det), 1)
    r = torch.matmul(u, vt)
    return r, biasq.squeeze()


# === Sequence Utilities ===
def Get_base(seq, basenpy_standard):
    base_num = basenpy_standard.shape[1]
    basenpy = np.zeros([len(seq), base_num, 3])
    seqnpy = np.array(list(seq))
    basenpy[seqnpy == "A"] = basenpy_standard[0]
    basenpy[seqnpy == "a"] = basenpy_standard[0]

    basenpy[seqnpy == "G"] = basenpy_standard[1]
    basenpy[seqnpy == "g"] = basenpy_standard[1]

    basenpy[seqnpy == "C"] = basenpy_standard[2]
    basenpy[seqnpy == "c"] = basenpy_standard[2]

    basenpy[seqnpy == "U"] = basenpy_standard[3]
    basenpy[seqnpy == "u"] = basenpy_standard[3]

    basenpy[seqnpy == "T"] = basenpy_standard[3]
    basenpy[seqnpy == "t"] = basenpy_standard[3]

    return torch.from_numpy(basenpy).double()
