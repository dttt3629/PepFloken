from pepfloken.modules.so3.utils import *

import logging
import os
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

def Omega(R): return torch.arccos((torch.diagonal(R, dim1=-2, dim2=-1).sum(axis=-1)-1)/2)

def f_igso3(omega, t, L=500):
    ls = torch.arange(L)[None]  
    return ((2*ls + 1) * torch.exp(-ls*(ls+1)*t/2) *
             torch.sin(omega[:, None]*(ls+1/2)) / torch.sin(omega[:, None]/2)).sum(dim=-1)

def igso3_density(Rt, t, L=500): return f_igso3(Omega(Rt), t, L)

def angle_density_unif(omega):
    return (1-torch.cos(omega))/np.pi

def tangent_gaussian(R0): return torch.einsum('...ij,...jk->...ik', R0, hat(torch.randn(R0.shape[0], 3)))

def centered_gaussian(num_batch, num_res, device='cpu'):
    noise = torch.randn(num_batch, num_res, 3, device=device)
    return noise - torch.mean(noise, dim=-2, keepdims=True)

def uniform_so3(num_batch, num_res, device='cpu'):
    return torch.tensor(
        Rotation.random(num_batch*num_res).as_matrix(),
        device=device,
        dtype=torch.float32,
    ).reshape(num_batch, num_res, 3, 3)

class SO3LookupCache:
    def __init__(
        self,
        cache_dir: str,
        cache_file: str,
        overwrite: bool = False,
    ) -> None:

        if not cache_file.endswith(".npz"):
            raise ValueError("Filename should have '.npz' extension.")
        self.cache_file = cache_file
        self.cache_dir = cache_dir
        self.cache_path = os.path.join(cache_dir, cache_file)
        self.overwrite = overwrite

    @property
    def path_exists(self) -> bool:
        return os.path.exists(self.cache_path)

    @property
    def dir_exists(self) -> bool:
        return os.path.exists(self.cache_dir)

    def delete_cache(self) -> None:

        if self.path_exists:
            os.remove(self.cache_path)

    def load_cache(self) -> Dict[str, torch.Tensor]:

        if self.path_exists:
            npz_data = np.load(self.cache_path)
            torch_dict = {f: torch.from_numpy(npz_data[f]) for f in npz_data.files}
            logger.info(f"Data loaded from {self.cache_path}")
            return torch_dict
        else:
            raise ValueError(f"No cache data found at {self.cache_path}.")

    def save_cache(self, data: Dict[str, torch.Tensor]) -> None:

        if not self.dir_exists:
            os.makedirs(self.cache_dir)

        if self.path_exists:
            if self.overwrite:
                logger.info("Overwriting cache ...")
                self.delete_cache()
            else:
                logger.warn(
                    f"Cache at {self.cache_path} exits and overwriting disabled. Doing nothing."
                )
        else:
            logger.info(f"Data saved to {self.cache_path}")
            numpy_dict = {k: v.detach().cpu().numpy() for k, v in data.items()}
            np.savez(self.cache_path, **numpy_dict)


class BaseSampleSO3(nn.Module):
    so3_type: str = "base"  

    def __init__(
        self,
        num_omega: int,
        sigma_grid: torch.Tensor,
        omega_exponent: int = 3,
        tol: float = 1e-7,
        interpolate: bool = True,
        cache_dir: Optional[str] = None,
        overwrite_cache: bool = False,
        device: str = 'cpu',
    ) -> None:

        super().__init__()
        self.num_omega = num_omega
        self.omega_exponent = omega_exponent
        self.tol = tol
        self.interpolate = interpolate
        self.device = device
        self.register_buffer("sigma_grid", sigma_grid, persistent=False)

        omega_grid, cdf_igso3 = self._setup_lookup(sigma_grid, cache_dir, overwrite_cache)
        self.register_buffer("omega_grid", omega_grid, persistent=False)
        self.register_buffer("cdf_igso3", cdf_igso3, persistent=False)

    def _setup_lookup(
        self,
        sigma_grid: torch.Tensor,
        cache_dir: Optional[str] = None,
        overwrite_cache: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if cache_dir is not None:
            cache_name = self._get_cache_name()
            cache = SO3LookupCache(cache_dir, cache_name, overwrite=True)

            if cache.path_exists and not overwrite_cache:
                cache_data = cache.load_cache()
                omega_grid = cache_data["omega_grid"]
                cdf_igso3 = cache_data["cdf_igso3"]
            else:
                omega_grid, cdf_igso3 = self._generate_lookup(sigma_grid)
                cache.save_cache({"omega_grid": omega_grid, "cdf_igso3": cdf_igso3})
        else:
            omega_grid, cdf_igso3 = self._generate_lookup(sigma_grid)

        return omega_grid.to(sigma_grid.dtype), cdf_igso3.to(sigma_grid.dtype)

    def _get_cache_name(self) -> str:

        cache_name = "cache_{:s}_s{:04.3f}-{:04.3f}-{:d}_o{:d}-{:d}.npz".format(
            self.so3_type,
            torch.min(self.sigma_grid).cpu().item(),
            torch.max(self.sigma_grid).cpu().item(),
            self.sigma_grid.shape[0],
            self.num_omega,
            self.omega_exponent,
        )
        return cache_name

    def get_sigma_idx(self, sigma: torch.Tensor) -> torch.Tensor:

        return torch.bucketize(sigma, self.sigma_grid)

    def expansion_function(
        self, omega_grid: torch.Tensor, sigma_grid: torch.Tensor
    ) -> torch.Tensor:

        raise NotImplementedError

    @torch.no_grad()
    def _generate_lookup(self, sigma_grid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:


        current_device = sigma_grid.device
        sigma_grid_tmp = sigma_grid.to(torch.float64)

        if torch.cuda.is_available():
            sigma_grid_tmp = sigma_grid_tmp.to(device=self.device)

        omega_grid = torch.linspace(0.0, 1, self.num_omega + 1).to(sigma_grid_tmp)

        omega_grid = omega_grid**self.omega_exponent

        omega_grid = omega_grid * np.pi

        pdf_igso3 = self.expansion_function(omega_grid, sigma_grid_tmp)

        pdf_igso3 = pdf_igso3 * (1.0 - torch.cos(omega_grid)) / np.pi

        cdf_igso3 = integrate_trapezoid_cumulative(pdf_igso3, omega_grid)
        cdf_igso3 = cdf_igso3 / cdf_igso3[:, -1][:, None]

        cdf_igso3 = cdf_igso3.to(device=current_device)
        omega_grid = omega_grid.to(device=current_device)

        return omega_grid[1:].to(sigma_grid.dtype), cdf_igso3.to(sigma_grid.dtype)

    def sample(self, sigma: torch.Tensor, num_samples: int) -> torch.Tensor:


        vectors = self.sample_vector(sigma.shape[0], num_samples)
        angles = self.sample_angle(sigma, num_samples)

        angles = self._process_angles(sigma, angles)

        rotation_vectors = vectors * angles[..., None]

        rotation_matrices = rotvec_to_rotmat(rotation_vectors, tol=self.tol)
        return rotation_matrices

    def _process_angles(self, sigma: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:

        return angles

    def sample_vector(self, num_sigma: int, num_samples: int) -> torch.Tensor:

        vectors = torch.randn(num_sigma, num_samples, 3, device=self.sigma_grid.device)
        vectors = vectors / torch.norm(vectors, dim=2, keepdim=True)
        return vectors

    def sample_angle(self, sigma: torch.Tensor, num_samples: int) -> torch.Tensor:

        sigma_indices = self.get_sigma_idx(sigma)
        cdf_tmp = self.cdf_igso3[sigma_indices, :]

        p_uniform = torch.rand((*sigma_indices.shape, *[num_samples]), device=sigma_indices.device)

        idx_stop = torch.sum(cdf_tmp[..., None] < p_uniform[:, None, :], dim=1).long()
        idx_start = torch.clamp(idx_stop - 1, min=0)

        if not self.interpolate:
            omega = torch.gather(cdf_tmp, dim=1, index=idx_stop)
        else:
            cdf_start = torch.gather(cdf_tmp, dim=1, index=idx_start)
            cdf_stop = torch.gather(cdf_tmp, dim=1, index=idx_stop)

            cdf_delta = torch.clamp(cdf_stop - cdf_start, min=self.tol)
            cdf_weight = torch.clamp((p_uniform - cdf_start) / cdf_delta, min=0.0, max=1.0)

            omega_start = self.omega_grid[idx_start]
            omega_stop = self.omega_grid[idx_stop]

            omega = torch.lerp(omega_start, omega_stop, cdf_weight)

        return omega


class SampleIGSO3(BaseSampleSO3):
    so3_type = "igso3"  

    def __init__(
        self,
        num_omega: int,
        sigma_grid: torch.Tensor,
        omega_exponent: int = 3,
        tol: float = 1e-7,
        interpolate: bool = True,
        l_max: int = 1000,
        cache_dir: Optional[str] = None,
        overwrite_cache: bool = False,
        device: str = 'cpu',
    ) -> None:

        self.l_max = l_max
        super().__init__(
            num_omega=num_omega,
            sigma_grid=sigma_grid,
            omega_exponent=omega_exponent,
            tol=tol,
            interpolate=interpolate,
            cache_dir=cache_dir,
            overwrite_cache=overwrite_cache,
            device=device,
        )

    def _get_cache_name(self) -> str:

        cache_name = "cache_{:s}_s{:04.3f}-{:04.3f}-{:d}_l{:d}_o{:d}-{:d}.npz".format(
            self.so3_type,
            torch.min(self.sigma_grid).cpu().item(),
            torch.max(self.sigma_grid).cpu().item(),
            self.sigma_grid.shape[0],
            self.l_max,
            self.num_omega,
            self.omega_exponent,
        )
        return cache_name

    def expansion_function(
        self,
        omega_grid: torch.Tensor,
        sigma_grid: torch.Tensor,
    ) -> torch.Tensor:

        return generate_igso3_lookup_table(omega_grid, sigma_grid, l_max=self.l_max, tol=self.tol)

    def _process_angles(self, sigma: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:

        angles = torch.where(
            sigma[..., None] < self.tol,
            torch.zeros_like(angles),
            angles,
        )
        return angles



class SampleUSO3:
    def sample(self, sigma: torch.Tensor, num_samples: int):
        return torch.tensor(Rotation.random(num_samples).as_matrix(), dtype=torch.float32)
















































@torch.no_grad()
def integrate_trapezoid_cumulative(f_grid: torch.Tensor, x_grid: torch.Tensor) -> torch.Tensor:

    f_sum = f_grid[..., :-1] + f_grid[..., 1:]
    delta_x = torch.diff(x_grid, dim=-1)
    integral = torch.cumsum((f_sum * delta_x[None, :]) / 2.0, dim=-1)
    return integral


def uniform_so3_density(omega: torch.Tensor) -> torch.Tensor:

    return (1.0 - torch.cos(omega)) / np.pi


def igso3_expansion(
    omega: torch.Tensor, sigma: torch.Tensor, l_grid: torch.Tensor, tol=1e-7
) -> torch.Tensor:

    denom_sin = torch.sin(0.5 * omega)

    l_fac_1 = 2.0 * l_grid + 1.0
    l_fac_2 = -l_grid * (l_grid + 1.0)

    numerator_sin = torch.sin((l_grid[None, :] + 1 / 2) * omega[:, None])

    exponential_term = l_fac_1[None, :] * torch.exp(l_fac_2[None, :] * sigma[:, None] ** 2 / 2)

    f_igso = torch.sum(exponential_term * numerator_sin, dim=1)
    f_limw = torch.sum(exponential_term * l_fac_1[None, :], dim=1)

    f_igso = f_igso / (denom_sin + tol)

    f_igso = torch.where(omega <= tol, f_limw, f_igso)

    f_igso = torch.where(
        torch.logical_or(torch.isinf(f_igso), torch.isnan(f_igso)), torch.zeros_like(f_igso), f_igso
    )

    return f_igso


def digso3_expansion(
    omega: torch.Tensor, sigma: torch.Tensor, l_grid: torch.Tensor, tol=1e-7
) -> torch.Tensor:

    denom_cos = 1.0 - torch.cos(omega)

    l_fac_1 = 2.0 * l_grid + 1.0
    l_fac_2 = l_grid + 1.0
    l_fac_3 = -l_grid * l_fac_2

    numerator_sin = l_grid[None, :] * torch.sin(l_fac_2[None, :] * omega[:, None]) - l_fac_2[
        None, :
    ] * torch.sin(l_grid[None, :] * omega[:, None])

    df_igso = torch.sum(
        l_fac_1[None, :] * torch.exp(l_fac_3[None, :] * sigma[:, None] ** 2 / 2) * numerator_sin,
        dim=1,
    )

    df_igso = df_igso / (denom_cos + tol)

    df_igso = torch.where(omega <= tol, torch.zeros_like(df_igso), df_igso)

    df_igso = torch.where(
        torch.logical_or(torch.isinf(df_igso), torch.isnan(df_igso)),
        torch.zeros_like(df_igso),
        df_igso,
    )

    return df_igso


def dlog_igso3_expansion(
    omega: torch.Tensor, sigma: torch.Tensor, l_grid: torch.Tensor, tol=1e-7
) -> torch.Tensor:

    f_igso3 = igso3_expansion(omega, sigma, l_grid, tol=tol)
    df_igso3 = digso3_expansion(omega, sigma, l_grid, tol=tol)

    return df_igso3 / (f_igso3 + tol)


@torch.no_grad()
def generate_lookup_table(
    base_function: Callable,
    omega_grid: torch.Tensor,
    sigma_grid: torch.Tensor,
    l_max: int = 1000,
    tol: float = 1e-7,
):

    l_grid = torch.arange(l_max + 1, device=omega_grid.device).to(omega_grid.dtype)

    n_omega = len(omega_grid)
    n_sigma = len(sigma_grid)

    f_table = torch.zeros(n_sigma, n_omega, device=omega_grid.device, dtype=omega_grid.dtype)

    for eps_idx in tqdm(range(n_sigma), desc=f"Computing {base_function.__name__}"):
        f_table[eps_idx, :] = base_function(
            omega_grid,
            torch.ones_like(omega_grid) * sigma_grid[eps_idx],
            l_grid,
            tol=tol,
        )

    return f_table


def generate_igso3_lookup_table(
    omega_grid: torch.Tensor,
    sigma_grid: torch.Tensor,
    l_max: int = 1000,
    tol: float = 1e-7,
) -> torch.Tensor:

    f_igso = generate_lookup_table(
        base_function=igso3_expansion,
        omega_grid=omega_grid,
        sigma_grid=sigma_grid,
        l_max=l_max,
        tol=tol,
    )
    return f_igso


def generate_dlog_igso3_lookup_table(
    omega_grid: torch.Tensor,
    sigma_grid: torch.Tensor,
    l_max: int = 1000,
    tol: float = 1e-7,
) -> torch.Tensor:

    dlog_igso = generate_lookup_table(
        base_function=dlog_igso3_expansion,
        omega_grid=omega_grid,
        sigma_grid=sigma_grid,
        l_max=l_max,
        tol=tol,
    )
    return dlog_igso


if __name__ == '__main__':
    sigma_grid = torch.linspace(0.1, 1.5, 1000)
    igso3 = SampleIGSO3(1000, sigma_grid, cache_dir='.cache')
    print(igso3.sample(torch.tensor([1.5]),4))

    uso3 = SampleUSO3(1000, sigma_grid, cache_dir='.cache')
    print(uso3.sample(torch.tensor([1.5]),4))