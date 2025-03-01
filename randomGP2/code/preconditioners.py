import torch
from linear_operator.utils.cholesky import psd_safe_cholesky

import torch
from gpytorch.functions import pivoted_cholesky
from torch import jit



import torch
from typing import Callable

import torch

def _pivoted_cholesky(x, kernel_fn, outputscale, rank=100):
    n = x.shape[0]
    L = torch.zeros((rank, n), device=x.device)
    d = (outputscale ** 2) * torch.ones(n, device=x.device)
    pi = torch.arange(n, device=x.device)
    
    for m in range(rank):
        # Find pivot: index of maximum value in d[m:]
        pivot_relative = torch.argmax(d[m:]).item()
        i = m + pivot_relative

        # Swap entries in pi at indices m and i
        temp = pi[m].clone()
        pi[m] = pi[i]
        pi[i] = temp

        # Set the m-th row pivot entry: L[m, pi[m]] = sqrt(d[pi[m]])
        L[m, pi[m]] = torch.sqrt(d[pi[m]])
        
        # Compute kernel evaluations between the pivot and the remaining points
        # Unsqueeze the pivot to match dimensions: from [features] to [1, features]
        a = kernel_fn(x[pi[m]].unsqueeze(0), x[pi[m+1:]]).squeeze(0)  # Expected shape: (n - m - 1,)
        
        # Compute correction from previous rows:
        if m > 0:
            correction = torch.sum(L[:m, pi[m]].unsqueeze(1) * L[:m, pi[m+1:]], dim=0)
        else:
            correction = 0.0

        # Compute the new elements for L's m-th row for the remaining indices
        l = (a - correction) / L[m, pi[m]]
        L[m, pi[m+1:]] = l

        # Update the residual diagonal values for the remaining indices
        d[pi[m+1:]] = d[pi[m+1:]] - (l ** 2)
    
    return L

@torch.jit.script
def identity_precon(x: torch.Tensor) -> torch.Tensor:
    return x


import torch

def build_cholesky(X: torch.Tensor, kernel: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], noise: float, rank: int):
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