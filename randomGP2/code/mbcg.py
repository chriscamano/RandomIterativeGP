
"""
Copyright (c) 2022, Wesley Maddox, Andres Potapczynski, Andrew Gordon Wilson
All rights reserved.

This file contains modifications to original binary 
"""
from gpytorch.utils.deprecation import bool_compat
import torch
import gpytorch

import torch
import torch

def build_lanczos_batched(alpha_history, beta_history, dtype, device, eps=1e-8):
    """
    Construct a batch of Lanczos tridiagonal matrices T from batched alpha and beta histories.

    Parameters:
        alpha_history (torch.Tensor): Tensor of shape (B, m) containing the alpha coefficients.
        beta_history (torch.Tensor): Tensor of shape (B, m-1) containing the beta coefficients.
        dtype: Torch data type to use.
        device: Torch device.
        eps (float): Small value to prevent numerical instability.

    Returns:
        torch.Tensor: A tensor of shape (B, m, m) containing the Lanczos tridiagonal matrices T for each batch.
    """
    B, m = alpha_history.shape

    # Clamp values to avoid division by zero
    alpha = torch.clamp(alpha_history, min=eps)
    beta = torch.clamp(beta_history, min=eps) if m > 1 else None

    # Compute diagonal elements: shape (B, m)
    # For the first element: 1/alpha_0
    diag0 = 1.0 / alpha[:, 0:1]  # shape (B, 1)
    if m > 1:
        # For i >= 1: 1/alpha_i + beta_{i-1}/alpha_{i-1}
        diag_rest = 1.0 / alpha[:, 1:] + beta / alpha[:, :-1]
        diag = torch.cat([diag0, diag_rest], dim=1)  # shape (B, m)
    else:
        diag = diag0

    # Construct the diagonal of T using batched diag_embed.
    T_batch = torch.diag_embed(diag)  # shape (B, m, m)

    # If m > 1, compute off-diagonals: shape (B, m-1)
    if m > 1:
        off_diag = (beta ** 0.5) / alpha[:, :-1]
        # Add the upper and lower diagonals
        T_batch = T_batch + torch.diag_embed(off_diag, offset=1) + torch.diag_embed(off_diag, offset=-1)

    return T_batch


def build_lanczos(alpha_list, beta_list, dtype, device, eps=1e-8):
    """
    Construct the Lanczos tridiagonal matrix T from lists of alpha and beta coefficients.

    Parameters:
        alpha_list (list of float): List containing the alpha coefficients.
        beta_list (list of float): List containing the beta coefficients.
        dtype: Torch data type to use.
        device: Torch device.
        eps (float): Small value to prevent numerical instability.

    Returns:
        torch.Tensor: The Lanczos tridiagonal matrix T.

    The matrix T is defined as:
    \[
    T_{1,1} = \frac{1}{\alpha_1}, \quad
    T_{i,i} = \frac{1}{\alpha_{i+1}}+\frac{\beta_i}{\alpha_i} \text{ for } i\ge1, \quad
    T_{i,i+1} = T_{i+1,i} = \frac{\sqrt{\beta_i}}{\alpha_i} \text{ for } i \ge 1.
    \]
    """
    m = len(alpha_list)
    T = torch.zeros((m, m), dtype=dtype, device=device)

    # Convert to tensors
    alpha_list = torch.tensor(alpha_list, dtype=dtype, device=device)
    beta_list = torch.tensor(beta_list, dtype=dtype, device=device)

    # Avoid division by zero by clamping very small values
    alpha_list = torch.clamp(alpha_list, min=eps)
    beta_list = torch.clamp(beta_list, min=eps)


    # Populate the Lanczos tridiagonal matrix
    T[0, 0] = 1.0 / alpha_list[0]
    for i in range(1, m):
        T[i, i] = 1.0 / alpha_list[i] + beta_list[i-1] / alpha_list[i-1]

    for i in range(m - 1):
        T[i, i+1] = (beta_list[i] ** 0.5) / alpha_list[i]
        T[i+1, i] = (beta_list[i] ** 0.5) / alpha_list[i]

    return T
    
def lanczos_decomp(K_train, p, device, dtype, tol=1e-16, num_reorth_vecs=None):
    """
    Compute a rank-p Lanczos decomposition of the symmetric positive-definite matrix K_train.
    
    The algorithm constructs an orthonormal basis Q and a tridiagonal matrix T such that:
    
        K_train ≈ Q T Q^T.
    
    Parameters:
        K_train (torch.Tensor): The training kernel matrix.
        p (int): The desired rank for the Lanczos decomposition.
        device: The torch device.
        dtype: The torch data type.
        tol (float): Tolerance for convergence.
        num_reorth_vecs (int, optional): Number of previous Lanczos vectors to reorthogonalize against.
                                         If None, reorthogonalize against all computed vectors (full reorthogonalization).
    
    Returns:
        Q (torch.Tensor): Orthonormal basis of shape (n, p) (or fewer columns if converged early).
        T (torch.Tensor): Tridiagonal matrix of shape (p, p) (or smaller if converged early).
    """
    n = K_train.shape[0]
    Q = torch.zeros(n, p, dtype=dtype, device=device)
    T = torch.zeros(p, p, dtype=dtype, device=device)
    beta = 0.0
    q_prev = torch.zeros(n, dtype=dtype, device=device)
    
    # Random vector of 2-norm unity (e.g., as in Saad 2003)
    q = torch.randn(n, dtype=dtype, device=device)
    q = q / q.norm()
    
    for j in range(p):
        Q[:, j] = q
        z = K_train @ q
        alpha = torch.dot(q, z)
        T[j, j] = alpha
        if j > 0:
            T[j, j - 1] = beta
            T[j - 1, j] = beta
        
        # Standard update
        z = z - alpha * q - beta * q_prev
        
        # Determine the subset of previous vectors to reorthogonalize against.
        if num_reorth_vecs is None:
            # Use all previously computed vectors.
            vec_subset = Q[:, :j+1]
        else:
            # Use only the most recent 'num_reorth_vecs' vectors.
            num = min(num_reorth_vecs, j+1)
            vec_subset = Q[:, (j+1-num):(j+1)]
        
        # Reorthogonalize z against the chosen subset.
        z = z - vec_subset @ (vec_subset.t() @ z)
        
        beta = z.norm()
        if beta < tol:  # Convergence check
            return Q[:, :j+1], T[:j+1, :j+1]
        q_prev = q
        q = z / beta
    return Q, T



def linear_cg(
    matmul_closure,
    rhs,
    tolerance=None,
    eps=1e-10,
    stop_updating_after=1e-10,
    max_iter=None,
    initial_guess=None,
    preconditioner=None,
):
    rhs_norm = rhs.norm(2, dim=-2, keepdim=True)
    rhs_is_zero = rhs_norm.lt(eps)
    rhs_norm = rhs_norm.masked_fill_(rhs_is_zero, 1)
    rhs = rhs.div(rhs_norm)

    state, out = initialize_cg(matmul_closure, rhs, stop_updating_after, eps)
    x0, has_converged, r0, batch_shape, residual_norm = state
    (p0, gamma0, mul_storage, beta, alpha, is_zero, z0) = out

    for k in range(max_iter):
        Ap0 = matmul_closure(p0)
        take_cg_step(
            Ap0=Ap0,
            x0=x0,
            r0=r0,
            gamma0=gamma0,
            p0=p0,
            alpha=alpha,
            beta=beta,
            z0=z0,
            mul_storage=mul_storage,
            has_converged=has_converged,
            eps=eps,
            is_zero=is_zero,
        )

        if cond_fn(k, max_iter, tolerance, r0, has_converged, residual_norm,
                   stop_updating_after, rhs_is_zero):
            break

    x0 = x0.mul(rhs_norm)
    return x0


def initialize_cg(matmul_closure, rhs, stop_updating_after, eps, preconditioner):
    initial_guess = torch.zeros_like(rhs)
    eps = torch.tensor(eps, dtype=rhs.dtype, device=rhs.device)

    residual = rhs - matmul_closure(initial_guess)
    batch_shape = residual.shape[:-2]

    result = initial_guess.expand_as(residual).contiguous()

    residual_norm = residual.norm(2, dim=-2, keepdim=True)
    has_converged = torch.lt(residual_norm, stop_updating_after)

    state = (result, has_converged, residual, batch_shape, residual_norm)
    out = create_placeholders(rhs, residual, preconditioner, batch_shape)
    return state, out

def take_cg_step(Ap0, x0, r0, gamma0, p0, alpha, beta, z0, mul_storage, has_converged, eps, is_zero, precon):
    dot = torch.sum(p0 * Ap0, dim=-2, keepdim=True)
    is_small = dot < eps
    alpha_new = gamma0 / torch.where(is_small, torch.ones_like(dot), dot)
    alpha_new = torch.where(is_small | has_converged, torch.zeros_like(alpha_new), alpha_new)
    r_new = r0 - alpha_new * Ap0
    x_new = x0 + alpha_new * p0
    precond_residual = precon(r_new)
    new_gamma = torch.sum(r_new * precond_residual, dim=-2, keepdim=True)
    is_small_gamma = gamma0 < eps
    beta_new = new_gamma / torch.where(is_small_gamma, torch.ones_like(gamma0), gamma0)
    beta_new = torch.where(is_small_gamma, torch.zeros_like(beta_new), beta_new)
    p_new = precond_residual + beta_new * p0
    return x_new, r_new, new_gamma, beta_new, p_new, alpha_new


def create_placeholders(rhs, residual, preconditioner, batch_shape):
    precond_residual = preconditioner(residual)
    curr_conjugate_vec = precond_residual
    residual_inner_prod = precond_residual.mul(residual).sum(-2, keepdim=True)

    mul_storage = torch.empty_like(residual)
    alpha = torch.empty(*batch_shape, 1, rhs.size(-1),
                        dtype=residual.dtype, device=residual.device)
    beta = torch.empty_like(alpha)
    is_zero = torch.empty(*batch_shape, 1, rhs.size(-1),
                          dtype=bool_compat, device=residual.device)
    return (curr_conjugate_vec, residual_inner_prod, mul_storage, beta, alpha, is_zero,
            precond_residual)



def cond_fn(k, max_iter, tolerance, residual, has_converged, residual_norm, stop_updating_after, rhs_is_zero):
    new_residual_norm = torch.norm(residual, 2, dim=-2, keepdim=True)
    new_residual_norm = torch.where(rhs_is_zero, torch.zeros_like(new_residual_norm), new_residual_norm)
    new_has_converged = new_residual_norm < stop_updating_after
    flag = k >= min(10, max_iter - 1) and bool(new_residual_norm.mean() < tolerance)
    return flag


def print_analysis(k, alpha, residual_norm, gamma0, beta):
    print('\n===================================================')
    print(f'Iter {k}')
    print(f'Residual norm mean: {torch.mean(residual_norm)}')
    print(f'Residual norm max: {torch.max(residual_norm)}')
    print(f'Residual norm: {residual_norm}')
    print('alpha')
    print(alpha)
    print(f'Alpha mean: {torch.mean(alpha)}')
    print('gamma')
    print(f'Gamma mean: {torch.mean(gamma0)}')
    print(gamma0)
    print('beta')
    print(f'Beta mean: {torch.mean(beta)}')
    print(beta)