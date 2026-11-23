import logging
import os
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import math

import warnings
warnings.filterwarnings('ignore')
from einops import rearrange

from copy import copy
from scipy.spatial.transform import Rotation

def gen_random_rotmats(num_samples):
    return torch.from_numpy(np.stack([Rotation.random().as_matrix() for _ in range(num_samples)])).float()

def gen_random_rotvecs(num_samples):
    return torch.from_numpy(np.stack([Rotation.random().as_rotvec() for _ in range(num_samples)])).float()

def check_skew_sym(h):
    SKEW_SYMMETRIC_TOL = 1e-4
    skew_sym_eq = torch.allclose(h, -h.permute(0, 2, 1), atol=SKEW_SYMMETRIC_TOL, rtol=0)
    if not skew_sym_eq:
        print('skew symmetric error', torch.abs(h+h.permute(0, 2, 1)).max())
        return False
    return True


def check_rot_mat(R):
    ROT_MAT_TOL = 1e-4
    rot_eq = torch.allclose(torch.inverse(R), R.permute(0, 2, 1), atol=ROT_MAT_TOL, rtol=0)
    rot_det_eq = torch.allclose(torch.det(R), torch.ones_like(torch.det(R)), atol=ROT_MAT_TOL, rtol=0)    
    if not (rot_eq and rot_det_eq):
        return False
    return True

def scale_rotmat(
    rotation_matrix: torch.Tensor, scalar: torch.Tensor, tol: float = 1e-7
) -> torch.Tensor:

    assert rotation_matrix.ndim - 1 == scalar.ndim
    scaled_rmat = rotvec_to_rotmat(rotmat_to_rotvec(rotation_matrix) * scalar, tol=tol)
    return scaled_rmat


def _broadcast_identity(target: torch.Tensor) -> torch.Tensor:

    id3 = torch.eye(3, device=target.device, dtype=target.dtype)
    id3 = torch.broadcast_to(id3, target.shape)
    return id3


def skew_matrix_exponential_map_axis_angle(
    angles: torch.Tensor, skew_matrices: torch.Tensor
) -> torch.Tensor:

    id3 = _broadcast_identity(skew_matrices)

    angles = angles[..., None, None]

    exp_skew = (
        id3
        + torch.sin(angles) * skew_matrices
        + (1.0 - torch.cos(angles))
        * torch.einsum("b...ik,b...kj->b...ij", skew_matrices, skew_matrices)
    )
    return exp_skew


def skew_matrix_exponential_map(
    angles: torch.Tensor, skew_matrices: torch.Tensor, tol=1e-7
) -> torch.Tensor:

    id3 = _broadcast_identity(skew_matrices)

    angles = angles[..., None, None]
    angles_sq = angles.square()

    sin_coeff = torch.sin(angles) / angles
    cos_coeff = (1.0 - torch.cos(angles)) / angles_sq
    sin_coeff_small = 1.0 - angles_sq / 6.0
    cos_coeff_small = 0.5 - angles_sq / 24.0

    mask_zero = torch.abs(angles) < tol
    sin_coeff = torch.where(mask_zero, sin_coeff_small, sin_coeff)
    cos_coeff = torch.where(mask_zero, cos_coeff_small, cos_coeff)

    exp_skew = (
        id3
        + sin_coeff * skew_matrices
        + cos_coeff * torch.einsum("b...ik,b...kj->b...ij", skew_matrices, skew_matrices)
    )
    return exp_skew


def rotvec_to_rotmat(rotation_vectors: torch.Tensor, tol: float = 1e-7) -> torch.Tensor:

    rotation_angles = torch.norm(rotation_vectors, dim=-1)

    skew_matrices = vector_to_skew_matrix(rotation_vectors)

    rotation_matrices = skew_matrix_exponential_map(rotation_angles, skew_matrices, tol=tol)

    return rotation_matrices


def rotmat_to_rotvec(rotation_matrices: torch.Tensor) -> torch.Tensor:

    angles, angles_sin, _ = angle_from_rotmat(rotation_matrices)
    vector = skew_matrix_to_vector(rotation_matrices - rotation_matrices.transpose(-2, -1))

    mask_zero = torch.isclose(angles, torch.zeros_like(angles)).to(angles.dtype)
    mask_pi = torch.isclose(angles, torch.full_like(angles, np.pi), atol=1e-2).to(angles.dtype)
    mask_else = (1 - mask_zero) * (1 - mask_pi)

    numerator = mask_zero / 2.0 + angles * mask_else
    denominator = (
        (1.0 - angles**2 / 6.0) * mask_zero  
        + 2.0 * angles_sin * mask_else  
        + mask_pi  
    )
    prefactor = numerator / denominator
    vector = vector * prefactor[..., None]

    id3 = _broadcast_identity(rotation_matrices)
    skew_outer = (id3 + rotation_matrices) / 2.0
    skew_outer = skew_outer + (torch.relu(skew_outer) - skew_outer) * id3

    vector_pi = torch.sqrt(torch.diagonal(skew_outer, dim1=-2, dim2=-1))

    signs_line_idx = torch.argmax(torch.norm(skew_outer, dim=-1), dim=-1).long()
    signs_line = torch.take_along_dim(skew_outer, dim=-2, indices=signs_line_idx[..., None, None])
    signs_line = signs_line.squeeze(-2)
    signs = torch.sign(signs_line)

    vector_pi = vector_pi * angles[..., None] * signs

    vector = vector + vector_pi * mask_pi[..., None]

    return vector


def angle_from_rotmat(
    rotation_matrices: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    skew_matrices = rotation_matrices - rotation_matrices.transpose(-2, -1)
    skew_vectors = skew_matrix_to_vector(skew_matrices)
    angles_sin = torch.norm(skew_vectors, dim=-1) / 2.0
    angles_cos = (torch.einsum("...ii", rotation_matrices) - 1.0) / 2.0

    angles = torch.atan2(angles_sin, angles_cos)

    return angles, angles_sin, angles_cos


def vector_to_skew_matrix(vectors: torch.Tensor) -> torch.Tensor:

    skew_matrices = torch.zeros((*vectors.shape, 3), device=vectors.device, dtype=vectors.dtype)

    skew_matrices[..., 2, 1] = vectors[..., 0]
    skew_matrices[..., 0, 2] = vectors[..., 1]
    skew_matrices[..., 1, 0] = vectors[..., 2]

    skew_matrices = skew_matrices - skew_matrices.transpose(-2, -1)

    return skew_matrices


def skew_matrix_to_vector(skew_matrices: torch.Tensor) -> torch.Tensor:

    vectors = torch.zeros_like(skew_matrices[..., 0])
    vectors[..., 0] = skew_matrices[..., 2, 1]
    vectors[..., 1] = skew_matrices[..., 0, 2]
    vectors[..., 2] = skew_matrices[..., 1, 0]
    return vectors


def _rotquat_to_axis_angle(
    rotation_quaternions: torch.Tensor, tol: float = 1e-7
) -> Tuple[torch.Tensor, torch.Tensor]:

    rotation_axes = rotation_quaternions[..., 1:]
    rotation_axes_norms = torch.norm(rotation_axes, dim=-1)

    rotation_angles = 2.0 * torch.atan2(rotation_axes_norms, rotation_quaternions[..., 0])

    rotation_axes = rotation_axes / (rotation_axes_norms[:, None] + tol)
    return rotation_angles, rotation_axes


def rotquat_to_rotvec(rotation_quaternions: torch.Tensor) -> torch.Tensor:

    rotation_angles, rotation_axes = _rotquat_to_axis_angle(rotation_quaternions)
    rotation_vectors = rotation_axes * rotation_angles[..., None]
    return rotation_vectors


def rotquat_to_rotmat(rotation_quaternions: torch.Tensor) -> torch.Tensor:

    rotation_angles, rotation_axes = _rotquat_to_axis_angle(rotation_quaternions)
    skew_matrices = vector_to_skew_matrix(rotation_axes * rotation_angles[..., None])
    rotation_matrices = skew_matrix_exponential_map(rotation_angles, skew_matrices)
    return rotation_matrices


def apply_rotvec_to_rotmat(
    rotation_matrices: torch.Tensor,
    rotation_vectors: torch.Tensor,
    tol: float = 1e-7,
) -> torch.Tensor:

    rmat_right = rotvec_to_rotmat(rotation_vectors, tol=tol)
    rmat_rotated = torch.einsum("...ij,...jk->...ik", rotation_matrices, rmat_right)
    return rmat_rotated


def rotmat_to_skew_matrix(mat: torch.Tensor) -> torch.Tensor:

    vec = rotmat_to_rotvec(mat)
    return vector_to_skew_matrix(vec)


def skew_matrix_to_rotmat(skew: torch.Tensor) -> torch.Tensor:

    vec = skew_matrix_to_vector(skew)
    return rotvec_to_rotmat(vec)


def hat(vector: torch.Tensor) -> torch.Tensor:

    return vector_to_skew_matrix(vector)

def vee(matrix: torch.Tensor) -> torch.Tensor:

    return skew_matrix_to_vector(matrix)  

def exp(matrix: torch.Tensor) -> torch.Tensor:

    return skew_matrix_to_rotmat(matrix)

def log(rotmat: torch.Tensor) -> torch.Tensor:

    return rotmat_to_skew_matrix(rotmat)

def rot_transpose(rotmat: torch.Tensor) -> torch.Tensor: 

    return rotmat.transpose(-2, -1)

def rot_mult(mat_1: torch.Tensor, mat_2: torch.Tensor) -> torch.Tensor:

    return torch.einsum("...ij,...jk->...ik", mat_1, mat_2)

def multidim_trace(mat: torch.Tensor) -> torch.Tensor:

    return torch.einsum("...ii->...", mat)

def geodesic_dist(mat_1: torch.Tensor, mat_2: torch.Tensor) -> torch.Tensor:

    A = rotmat_to_skew_matrix(rot_mult(rot_transpose(mat_1), mat_2))
    return torch.sqrt(multidim_trace(rot_mult(A, rot_transpose(A))))

def expmap(tangent_vec: torch.Tensor, base_point: torch.Tensor) -> torch.Tensor:

    lie_vec = rot_mult(rot_transpose(base_point), tangent_vec)
    return rot_mult(base_point, exp(lie_vec))

def logmap(point: torch.Tensor, base_point: torch.Tensor) -> torch.Tensor:

    lie_point = rot_mult(rot_transpose(base_point), point)
    return rot_mult(base_point, log(lie_point))

DEFAULT_ACOS_BOUND: float = 1.0 - 1e-4

def acos_linear_extrapolation(
    x: torch.Tensor,
    bounds: Tuple[float, float] = (-DEFAULT_ACOS_BOUND, DEFAULT_ACOS_BOUND),
) -> torch.Tensor:


    lower_bound, upper_bound = bounds

    if lower_bound > upper_bound:
        raise ValueError("lower bound has to be smaller or equal to upper bound.")

    if lower_bound <= -1.0 or upper_bound >= 1.0:
        raise ValueError("Both lower bound and upper bound have to be within (-1, 1).")

    acos_extrap = torch.empty_like(x)
    x_upper = x >= upper_bound
    x_lower = x <= lower_bound
    x_mid = (~x_upper) & (~x_lower)

    acos_extrap[x_mid] = torch.acos(x[x_mid])
    acos_extrap[x_upper] = _acos_linear_approximation(x[x_upper], upper_bound)
    acos_extrap[x_lower] = _acos_linear_approximation(x[x_lower], lower_bound)

    return acos_extrap

def _acos_linear_approximation(x: torch.Tensor, x0: float) -> torch.Tensor:

    return (x - x0) * _dacos_dx(x0) + math.acos(x0)


def _dacos_dx(x: float) -> float:

    return (-1.0) / math.sqrt(1.0 - x * x)

def so3_relative_angle(
    R1: torch.Tensor,
    R2: torch.Tensor,
    cos_angle: bool = False,
    cos_bound: float = 1e-4,
    eps: float = 1e-4,
) -> torch.Tensor:

    R12 = R1.double() @ R2.permute(0, 2, 1).double()
    return so3_rotation_angle(R12, cos_angle=cos_angle, cos_bound=cos_bound, eps=eps)

def so3_rotation_angle(
    R: torch.Tensor,
    eps: float = 1e-4,
    cos_angle: bool = False,
    cos_bound: float = 1e-4,
) -> torch.Tensor:


    N, dim1, dim2 = R.shape
    if dim1 != 3 or dim2 != 3:
        raise ValueError("Input has to be a batch of 3x3 Tensors.")

    rot_trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    if ((rot_trace < -1.0 - eps) + (rot_trace > 3.0 + eps)).any():
        raise ValueError("A matrix has trace outside valid range [-1-eps,3+eps].")

    phi_cos = (rot_trace - 1.0) * 0.5

    if cos_angle:
        return phi_cos
    else:
        if cos_bound > 0.0:
            bound = 1.0 - cos_bound
            return acos_linear_extrapolation(phi_cos, (-bound, bound))
        else:
            return torch.acos(phi_cos)

def tangent_space_proj(base_point: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:

    skew_symmetric_part = 0.5 * (matrix - rot_transpose(matrix))

    return rot_mult(base_point, skew_symmetric_part)


def norm_SO3(base_point: torch.Tensor, tangent_vec: torch.Tensor) -> torch.Tensor:

    r = rot_mult(rot_transpose(base_point), tangent_vec)       
    norm = -torch.diagonal(r@r, dim1=-2, dim2=-1).sum(dim=-1)/2 
    return norm

def norm_SO3_aa(base_point: torch.Tensor, tangent_vec: torch.Tensor) -> torch.Tensor:

    r = rot_mult(rot_transpose(base_point), tangent_vec)       
    r_aa = skew_matrix_to_vector(r)          
    norm = torch.linalg.norm(r_aa, dim=-1)**2
    return norm

def geodesic_t(t: float, mat: torch.Tensor, base_mat: torch.Tensor, rot_vf=None) -> torch.Tensor:

    if rot_vf is None:
        rot_vf = rotmat_to_rotvec(rot_mult(rot_transpose(base_mat), mat))
    mat_t = rotvec_to_rotmat(t * rot_vf)
    if base_mat.shape != mat_t.shape:
        raise ValueError(
            f'Incompatible shapes: base_mat={base_mat.shape}, mat_t={mat_t.shape}')
    return torch.einsum("...ij,...jk->...ik", base_mat, mat_t)


def pairwise_geodesic_distance(x0, x1):

    batch_size = x0.size(0)
    x0 = rearrange(x0, 'b c d -> b (c d)', c=3, d=3)
    x1 = rearrange(x1, 'b c d -> b (c d)', c=3, d=3)
    mega_batch_x0 = rearrange(x0.repeat_interleave(batch_size, dim=0), 'b (c d) -> b c d', c=3, d=3)
    mega_batch_x1 = rearrange(x1.repeat(batch_size, 1), 'b (c d) -> b c d', c=3, d=3)
    distances = so3_relative_angle(mega_batch_x0, mega_batch_x1)**2
    return distances.reshape(batch_size, batch_size)



def calc_rot_vf(mat_t: torch.Tensor, mat_1: torch.Tensor) -> torch.Tensor:

    return rotmat_to_rotvec(rot_mult(rot_transpose(mat_t), mat_1))


def rotation_matrix_cosine_loss(R_pred, R_true):

    size = list(R_pred.shape[:-2])
    ncol = R_pred.numel() // 3

    RT_pred = R_pred.transpose(-2, -1).reshape(ncol, 3) 
    RT_true = R_true.transpose(-2, -1).reshape(ncol, 3) 

    ones = torch.ones([ncol, ], dtype=torch.long, device=R_pred.device)
    loss = F.cosine_embedding_loss(RT_pred, RT_true, ones, reduction='none')  
    loss = loss.reshape(size + [3]).sum(dim=-1)    
    return loss

if __name__ == "__main__":
    pass
