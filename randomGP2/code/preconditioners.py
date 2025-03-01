import torch
from linear_operator.utils.cholesky import psd_safe_cholesky

import torch
# from gpytorch.functions import pivoted_cholesky
from torch import jit



import torch
from typing import Callable

import torch

import torch



import torch

import torch
import torch

def _pivoted_cholesky(x, kernel_fn, outputscale, rank=100, epsilon=1e-8, jitter=1e-6):
    n = x.shape[0]
    L = torch.zeros((rank, n), device=x.device, dtype=x.dtype)
    d = outputscale**2 * torch.ones(n, device=x.device, dtype=x.dtype)
    pi = torch.arange(n, device=x.device)

    for m in range(rank):
        # Find pivot: index of maximum value in d[m:]
        pivot_relative = torch.argmax(d[m:]).item()
        i = m + pivot_relative

        # Swap entries in pi; clone if needed to break dependencies
        if i != m:
            pi = pi.clone()
            temp = pi[m].clone()
            pi[m] = pi[i]
            pi[i] = temp

        # Use clamped value for the pivot to avoid sqrt of negative/zero
        pivot_val = torch.clamp(d[pi[m]], min=epsilon)
        denom = torch.sqrt(pivot_val)
        L[m, pi[m]] = denom

        # Compute kernel evaluations between the pivot and the remaining points
        a = kernel_fn(x[pi[m]].unsqueeze(0), x[pi[m+1:]]).squeeze(0)
        
        # Compute correction from previous rows, if any
        if m > 0:
            temp1 = L[:m, pi[m]].clone()
            temp2 = L[:m, pi[m+1:]].clone()
            correction = torch.sum(temp1.unsqueeze(1) * temp2, dim=0)
        else:
            correction = 0.0

        # Compute new elements for L's m-th row for the remaining indices
        l = (a - correction) / denom
        L[m, pi[m+1:]] = l

        # Update d for the remaining indices using an out-of-place update
        d_updated = d[pi[m+1:]] - (l ** 2)
        # Clamp to ensure non-negativity and add jitter
        d = d.index_copy(0, pi[m+1:], torch.clamp(d_updated, min=epsilon) + jitter)
    
    return L

@torch.jit.script
def identity_precon(x: torch.Tensor) -> torch.Tensor:
    return x


import torch

def build_cholesky(X: torch.Tensor, kernel: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], noise: float, rank: int):
    # L=pivoted_cholesky(X, rank, error_tol=1e-14)
    L = _pivoted_cholesky(X, kernel, kernel.outputscale, rank=rank)  # L has shape (rank, n)
    noise_inv2 = noise ** -2
    noise_inv4 = noise ** -4
    M = torch.eye(rank, device=L.device, dtype=L.dtype) + noise_inv2 * (L @ L.T)
    M_cho_factor = psd_safe_cholesky(M)
    
    @torch.jit.ignore
    def precond_inv(v: torch.Tensor) -> torch.Tensor:
        if v.ndim == 1:
            v = v.unsqueeze(1)
        tmp = noise_inv2 * (L.T @ torch.cholesky_solve(L @ v, M_cho_factor, upper=False))
        result = noise_inv2 * (v - tmp)
        return result.squeeze(-1) if result.ndim == 2 and result.shape[1] == 1 else result

    preconditioner_matrix = L @ L.T + noise**2 * torch.eye(rank, device=L.device, dtype=L.dtype)
    
    return precond_inv, preconditioner_matrix, L.T