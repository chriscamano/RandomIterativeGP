import torch
import torch.nn as nn
import torch.nn.functional as F
from gpytorch.kernels import ScaleKernel
from abc import ABC, abstractmethod
import gpytorch
import os
import sys
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
if src_path not in sys.path:
    sys.path.append(src_path)
from linear_operator.utils.cholesky import psd_safe_cholesky

from gpytorch.functions import pivoted_cholesky
from mbcg import *
from mbcg import init_cg
from preconditioners import build_cholesky,identity_precon
import torch
import torch.nn as nn
import gpytorch
from abc import ABC, abstractmethod
import math 

import torch
from gpytorch.lazy import lazify
from gpytorch.functions import pivoted_cholesky,inv_quad_logdet,logdet
from linear_operator.functions._inv_quad_logdet import InvQuadLogdet

# from rpcholesky import simple_rpcholesky, accelerated_rpcholesky, block_rpcholesky, greedy, block_greedy
# from cg import my_cg_torch
# from rnla import  nyst_approximation_torch,logK_matvec_oracle,hutchplusplus

class Positive(nn.Module):
    def __init__(self, initial_value=1e-3, lower_bound=1e-3):
        super().__init__()
        self.lower_bound = lower_bound
        # Compute initial raw value such that lower_bound + softplus(u) equals initial_value.
        if initial_value > lower_bound:
            u_init = torch.log(torch.exp(torch.tensor(initial_value - lower_bound)) - 1.0)
        else:
            u_init = torch.tensor(0.0)
        self.u = nn.Parameter(u_init)
    
    def get_value(self):
        return self.lower_bound + F.softplus(self.u)

class AbstractGaussianProcess(nn.Module, ABC):
    def __init__(self, kernel, noise=0.1, dtype=torch.float, device="cuda:0",compute_covariance=True):
        super().__init__()
        self.dtype = dtype
        self.compute_covariance = compute_covariance
        self.device = device
        # Move kernel to the desired device and dtype.
        self.kernel = kernel.to(dtype=dtype, device=device)
        
        # Use the Positive constraint for the noise parameter.
        self.noise = Positive(initial_value=noise)
        
        self.X_train = None
        self.alpha = None
        self.L = None

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass


class CholeskyGaussianProcess(AbstractGaussianProcess):    
    def fit(self, X, y):
        self.X_train = X.to(dtype=self.dtype, device=self.device)
        y = y.to(dtype=self.dtype, device=self.device)
        K = self.kernel(X, X).evaluate() + self.noise.get_value()**2 * torch.eye(len(X), dtype=self.dtype, device=self.device,)
  
        self.L = torch.linalg.cholesky(K)
        self.alpha = torch.cholesky_solve(y.unsqueeze(-1), self.L)

    def predict(self, X):
        if self.X_train is None or self.alpha is None:
            raise ValueError("Model must be trained before making predictions.")
        
        X = X.to(dtype=self.dtype, device=self.device)
        K_trans = self.kernel(X, self.X_train).to_dense()
        K = self.kernel(X, X)
        if self.compute_covariance:
            mean = K_trans.matmul( self.alpha)
            v = torch.cholesky_solve(K_trans.t(), self.L)
            covariance = K - K_trans @ v
            return mean.detach().squeeze(-1), covariance.detach().squeeze(-1)
        else:
            return mean.detach().squeeze(-1)

    def compute_mll(self, y,mll_debug=False): #Exact MLL computation
        n = y.shape[0]
        log_det_K = 2 * torch.sum(torch.log(torch.diagonal(self.L)))
        quadratic_term = torch.dot(y.squeeze(), self.alpha.squeeze())
        const =n * torch.log(torch.tensor(2 * torch.pi, dtype=self.dtype, device=self.device))
        mll = 0.5 * (const + log_det_K + quadratic_term)
        if mll_debug:
            print("quadratic term Chol",quadratic_term,"log det Chol",log_det_K)
            return mll,quadratic_term,log_det_K
        else:
            return mll


import torch

class IterativeGaussianProcess(AbstractGaussianProcess):
    def __init__(self, kernel, noise=0.1, dtype=torch.float32, device="cuda:0",
                 cg_tol=1e-5, cg_max_iter=1000, warm_start=False, num_probes=16,
                 precon_type="identity", trace_backend="Hutch",
                 verbose=False, track_iterations=False,precon_rank=15,pred_lanczos_rank=None,
                 compute_covariance=True):
        super().__init__(kernel, noise, dtype, device)

        # self.grad_kernel = grad_kernel
        self.K_train=None
        #-------- CG parameters -------- 
        self.cg_tol = cg_tol
        self.cg_max_iter = cg_max_iter
        self.warm_start = warm_start
        self.num_probes = num_probes
        self.Z = None
        self.pinv_closure = None
        self.precon_matrix = None
        self.precon_type = precon_type
        self.previous_solutions = None
        self.precon_rank = precon_rank
        self.lanczos_iterates = []
        
        #-------- MLL parameters -------- 
        self.trace_backend = trace_backend

        #-------- Fast Predictive Covariance parameters -------- 
        self.pred_lanczos_rank = pred_lanczos_rank
        self.compute_covariance = compute_covariance
        self.F = None

        #-------- Tracking parameters -------- 
        self.verbose = verbose
        self.track_iterations = track_iterations
        if self.track_iterations:
            self.initialize_trackers()

    def get_preconditioner(self, X):
        if self.precon_type == "identity":
            self.pinv_closure = identity_precon
            n = X.shape[0]

            #TODO:add basecase identity grad computation O(1)
            ##self.preconditioner_matrix = torch.eye(n, dtype=self.dtype, device=self.device)
            #self.preconditioner_matrix.requires_grad_(True)  # Ensure it tracks gradients   
        elif self.precon_type == "piv_chol":
            # build_cholesky now returns (precond_inv, preconditioner_matrix, L_k)
            self.pinv_closure, self.preconditioner_matrix, self.L_k = build_cholesky(
                X, self.kernel, self.noise.get_value(), self.precon_rank
            )
            self.preconditioner_matrix.requires_grad_(True)  # Ensure it tracks gradients

        else:
            raise NotImplementedError(f"Preconditioner '{self.precon_type}' is not implemented.")


    def initialize_trackers(self):
        self.Us = []
        self.Rs = []
        self.gammas = []
        self.ps = []
        self.k = -1

    def update_trackers(self, x0, r0, gamma0, p0, k):
        if self.track_iterations:
            self.Us.append(x0.clone().cpu())
            self.Rs.append(r0.clone().cpu())
            self.gammas.append(gamma0.clone().cpu())
            self.ps.append(p0.clone().cpu())
            self.k = k

    def fit(self, X, y):
        self.X_train = X.to(dtype=self.dtype, device=self.device)
        y = y.to(dtype=self.dtype, device=self.device)
        n = self.X_train.shape[0]

        # Precompute the kernel matrix once and cache it
        # with torch.no_grad():

        self.get_preconditioner(self.X_train)
        self.K_lazy = self.kernel(self.X_train, self.X_train)
        def matmul_closure(b):
            return self.K_lazy.matmul(b) + self.noise.get_value()**2 * b


        #-------- Setup RHS of mbCG -------- 
        b = torch.empty((n, self.num_probes + 1), dtype=self.dtype, device=self.device)
        b[:, 0] = y.view(-1)

        if self.num_probes > 0:
            if self.precon_type == "piv_chol":
                k = self.L_k.shape[1]
                # Sample z1 ~ NORMAL(0, I_k) and z2 ~ NORMAL(0, I_n)
                z1 = torch.randn((k, self.num_probes), dtype=self.dtype, device=self.device)
                z2 = torch.randn((n, self.num_probes), dtype=self.dtype, device=self.device)
                sigma = self.noise.get_value()
                # Form random probe vectors via reparameterization: L_k @ z1 + sigma * z2
                rand_vectors = self.L_k @ z1 + sigma * z2
            else:
                rand_vectors = torch.randn(n, self.num_probes, dtype=self.dtype, device=self.device)
            
            #---- Normalize to lie on the unit sphere ----
            norms = torch.norm(rand_vectors, p=2, dim=0, keepdim=True)
            rand_vectors = rand_vectors / (norms + 1e-10)
            b[:, 1:] = rand_vectors

        # Normalize each column of b to unit norm
        rhs_norm = torch.norm(b, p=2, dim=0, keepdim=True)
        rhs_is_zero = rhs_norm.lt(1e-10)
        rhs_norm = rhs_norm.masked_fill(rhs_is_zero, 1)
        b_normalized = b / rhs_norm

        self.Z = b[:, 1:]
        #---- Warm start initialization ----
        if self.warm_start and self.previous_solutions is not None:
            x0 = self.previous_solutions
        else:
            x0 = None  # This triggers a zeros initial guess inside initialize_cg
        #---- Build preconditioner/state ----
        state, aux = init_cg(
            matmul_closure, b_normalized, stop_updating_after=1e-10,
            eps=1e-10, preconditioner=self.pinv_closure, initial_guess=x0
        )
        if self.track_iterations and not hasattr(self, 'Us'):
            self.initialize_trackers()

        x0, has_converged, r0, batch_shape, residual_norm = state
        (p0, gamma0, mul_storage, beta, alpha, is_zero, z0) = aux
        if self.verbose:
            self.update_trackers(x0, r0, gamma0, p0, k=0)

        #-------- CG iteration --------
        alpha_history_list = []
        beta_history_list = []
        iter_idx = 0
        #print("Training data range:", torch.min(self.X_train).item(), torch.max(self.X_train).item())
        #print("Any NaNs in training data?", torch.isnan(self.X_train).any().item())
        
        for k in range(1, self.cg_max_iter):
            #print("iter",k)
            #print(self.kernel(self.X_train,self.X_train).to_dense())

            Ap0 = matmul_closure(p0)
           # if torch.isnan(Ap0).any():
            #    print("Found NaNs in kernel matrix")
            x0, r0, gamma0, beta, p0, alpha = take_cg_step(
                Ap0, x0, r0, gamma0, p0, alpha, beta, z0, mul_storage,
                has_converged, eps=1e-10, is_zero=is_zero, precon=self.pinv_closure
            )
            # Accumulate the sliced vectors (ignoring the first column)
            alpha_history_list.append(alpha[0, 1:])
            beta_history_list.append(beta[0, 1:])
            iter_idx += 1

            if cond_fn(k, self.cg_max_iter, self.cg_tol, r0, has_converged,
                    residual_norm, stop_updating_after=1e-10, rhs_is_zero=rhs_is_zero):
                break
            if self.verbose:
                print_analysis(k, alpha, residual_norm, gamma0, beta)
                self.update_trackers(x0, r0, gamma0, p0, k)

        alpha_history = torch.stack(alpha_history_list, dim=0)
        beta_history = torch.stack(beta_history_list, dim=0)

       
        alpha_batch = alpha_history.transpose(0, 1)  # shape: (num_probes, iter_idx)
        beta_batch = beta_history.transpose(0, 1)    # shape: (num_probes, iter_idx)
        beta_batch = beta_batch[:, :alpha_batch.size(1) - 1]  # shape: (num_probes, iter_idx - 1)
        T_batch = build_lanczos_batched(alpha_batch, beta_batch, dtype=self.dtype, device=self.device, eps=1e-16)
        self.lanczos_iterates = [T_batch[i] for i in range(T_batch.size(0))]

        for i in range(T_batch.size(0)):
            # Grab the i-th tridiagonal matrix from the batch
            T_slice = T_batch[i]
            # Check for NaNs or Infs
            if torch.isnan(T_slice).any() or torch.isinf(T_slice).any():
                print(f"[WARNING] T_batch[{i}] contains NaNs or Infs!")
                print("Offending slice:", T_slice)
        solution = x0 * rhs_norm
        self.previous_solution = solution  # Cache for warm start
        self.alpha = solution[:, 0]
        self.probe_solutions = solution[:, 1:]


    def predict(self, X):
        X = X.to(dtype=self.dtype, device=self.device)
        K_trans = self.kernel(self.X_train, X)
        predictive_mean = K_trans.t() @ self.alpha  
        
        if self.compute_covariance:
            # If F has not been computed yet, compute it using Lanczos decomposition.
            if self.F is None:
                # if self_K_train == None:
                K_train = self.kernel(self.X_train, self.X_train).evaluate()
                K_train = K_train+ self.noise.get_value()**2
                K_train = K_train.detach()
                Q, T = lanczos_decomp(K_train, self.pred_lanczos_rank, self.device, self.dtype)
                L = psd_safe_cholesky(T)
                F_t = torch.linalg.solve_triangular(L, Q.t(), upper=False)
                self.F = F_t.t()
            K_test_full = self.kernel(X, X).evaluate()
            beta = self.F.t() @ K_trans
            covariance = K_test_full - beta.t() @ beta
            variance = torch.diag(covariance)
            return predictive_mean, covariance#, variance
        else:
            return predictive_mean,0

  

    def compute_mll(self, y,mll_debug=False):
        """
        Compute model marginal log-likelihood using Lanczos quadrature.
        Based on https://arxiv.org/pdf/2107.00243
        """
        y = y.to(dtype=self.dtype, device=self.device)
        n = y.shape[0]
        
        # Lanczos Quadrature
        if self.precon_type=="identity":
            tau_P_log = 0.0
        else:
            tau_P_log = torch.logdet(self.preconditioner_matrix)
        
        T_stack = torch.stack(self.lanczos_iterates, dim=0)
        T_stack = 0.5 * (T_stack + T_stack.transpose(-1, -2))
       # Check for NaNs or Infs before the eigendecomposition
        if torch.isnan(T_stack).any() or torch.isinf(T_stack).any():
            print("NaNs or Infs in T_stack!")
        eigenvals, eigenvecs = torch.linalg.eigh(T_stack)
        # eigenvals = torch.clamp(eigenvals, min=1e-16)
        
        eigenvals = torch.clamp(eigenvals, min=1e-16)
        weights = eigenvecs[:, 0, :] ** 2
        e1_entries = eigenvecs[:, 0, :]  
        e1_abs = e1_entries.abs()     
        weights = e1_abs**2             

        gamma = torch.sum(weights * torch.log(eigenvals), dim=1)
        gamma_sum = torch.sum(gamma)
        tau_star_log = tau_P_log + (n / self.num_probes) * gamma_sum
        
        quadratic = y.T @ self.alpha
        const_term = n * torch.log(torch.tensor(2 * torch.pi, dtype=self.dtype, device=self.device))
        
        mll = 0.5 * (quadratic + tau_star_log + const_term)
        if mll_debug:
            print("quadratic term igp",quadratic,"log det igp",tau_star_log)
            return mll,quadratic,tau_star_log
        else:
            return mll

    def estimate_mll_gradient(self):
        """
        Refactored to highlight three distinct areas of gradient computation:
        1) Noise gradient
        2) Outputscale gradient
        3) Lengthscale gradient
        
        Now the noise gradient is also obtained via autograd, 
        just like the other kernel parameters.
        """

        # Common setup and intermediate computations
        kernel = self.kernel
        n = self.X_train.shape[0]
        u0 = self.alpha.view(-1, 1)      # shape: (n, 1)
        U = self.probe_solutions        # shape: (n, t)
        Z = self.Z                      # shape: (n, t)

        if self.precon_matrix is None:
            P_inv = torch.eye(n, device=self.device)
        else:
            # If you rely on inverse, watch for ill-conditioning or large n
            P_inv = torch.linalg.inv(self.precon_matrix)

        u0_flat = u0.reshape(-1)
        u0_outer = torch.outer(u0_flat, u0_flat)  # shape: (n, n)

        # Build sum of outer products for the "summed_outer" term
        P_inv_Z = torch.matmul(P_inv, Z)  # shape: (n, t)
        t = Z.shape[1]  # number of probe vectors
        outer_products = torch.zeros((t, n, n), device=self.device, dtype=self.dtype)
        for i in range(t):
            outer_products[i] = torch.outer(P_inv_Z[:, i], U[:, i])
        summed_outer = outer_products.sum(dim=0) / t  # shape: (n, n)

        # Make sure the kernel parameters require grad
        for name, param in kernel.named_parameters():
            param.requires_grad_(True)

        # Build the training covariance K
        self.K_train = kernel(self.X_train, self.X_train).evaluate()
        # noise_value is softplus(noise.u), so K depends on noise.u
        K = self.K_train + self.noise.get_value()**2 * torch.eye(
            n, device=self.device, dtype=self.dtype
        )

        # Prepare to hold all gradients in a dict
        param_grads = {}

        # Gather kernel parameters (which might include raw_lengthscale etc.)
        params_list = list(kernel.parameters())
        param_names = [name for name, _ in kernel.named_parameters()]

        # 1) Kernel Parameter Gradients
        # We'll get partial derivatives of K wrt each kernel parameter by specifying
        # grad_outputs = (u0_outer) and then separately (-summed_outer).
        grads_term1 = torch.autograd.grad(
            outputs=K,
            inputs=params_list,
            grad_outputs=u0_outer,
            retain_graph=True,
            create_graph=False
        )
        grads_term2 = torch.autograd.grad(
            outputs=K,
            inputs=params_list,
            grad_outputs=-summed_outer,
            retain_graph=True,
            create_graph=False
        )

        # Combine them with the -0.5 factor, just like in your original approach
        for name, grad1, grad2 in zip(param_names, grads_term1, grads_term2):
            if name != "raw_outputscale":
                param_grads[name] = -0.5 * (grad1 + grad2)

        # 2) Outputscale gradient (manual example, as in your code)
        if "raw_outputscale" in param_names:
            idx_os = param_names.index("raw_outputscale")
            base_kernel = K / kernel.outputscale
            dT = torch.sigmoid(kernel.raw_outputscale)
            combined = (u0_outer - summed_outer) * base_kernel
            manual_grad = -0.5 * dT * torch.sum(combined)
            param_grads["raw_outputscale"] = manual_grad

        # 3) Noise gradient via autograd
        # Instead of manually computing noise_grad_estimate, we do exactly
        # the same pattern as for the kernel params. We call autograd.grad
        # on K with grad_outputs = (u0_outer) and then (-summed_outer),
        # collecting the partial derivative wrt self.noise.u (the raw noise param).

        grads_noise_term1 = torch.autograd.grad(
            outputs=K,
            inputs=[self.noise.u],
            grad_outputs=u0_outer,
            retain_graph=True,
            create_graph=False
        )[0]  # returns a 1-tuple
        grads_noise_term2 = torch.autograd.grad(
            outputs=K,
            inputs=[self.noise.u],
            grad_outputs=-summed_outer,
            retain_graph=False,  # no subsequent grads needed
            create_graph=False
        )[0]

        # Combine them with the same factor -0.5
        param_grads["noise"] = -0.5 * (grads_noise_term1 + grads_noise_term2)

        return param_grads




class IterativeGaussianProcess2(AbstractGaussianProcess):
    def __init__(self, kernel, noise=0.1, dtype=torch.float32, device="cuda:0",
                 cg_tol=1e-5, cg_max_iter=1000, warm_start=False, num_probes=16,
                 precon_type="identity", trace_backend="Hutch",
                 verbose=False, track_iterations=False,precon_rank=15,pred_lanczos_rank=None,
                 compute_covariance=True):
        super().__init__(kernel, noise, dtype, device)

        # self.grad_kernel = grad_kernel
        self.K_train=None
        #-------- CG parameters -------- 
        self.cg_tol = cg_tol
        self.cg_max_iter = cg_max_iter
        self.warm_start = warm_start
        self.num_probes = num_probes
        self.Z = None
        self.pinv_closure = None
        self.precon_matrix = None
        self.precon_type = precon_type
        self.previous_solutions = None
        self.precon_rank = precon_rank
        self.lanczos_iterates = []
        
        #-------- MLL parameters -------- 
        self.trace_backend = trace_backend

        #-------- Fast Predictive Covariance parameters -------- 
        self.pred_lanczos_rank = pred_lanczos_rank
        self.compute_covariance = compute_covariance
        self.F = None

        #-------- Tracking parameters -------- 
        self.verbose = verbose
        self.track_iterations = track_iterations
        if self.track_iterations:
            self.initialize_trackers()

    def get_preconditioner(self, X):
        if self.precon_type == "identity":
            self.pinv_closure = identity_precon
            n = X.shape[0]

            #TODO:add basecase identity grad computation O(1)
            ##self.preconditioner_matrix = torch.eye(n, dtype=self.dtype, device=self.device)
            #self.preconditioner_matrix.requires_grad_(True)  # Ensure it tracks gradients   
        elif self.precon_type == "piv_chol":
            # build_cholesky now returns (precond_inv, preconditioner_matrix, L_k)
            self.pinv_closure, self.preconditioner_matrix, self.L_k = build_cholesky(
                X, self.kernel, self.noise.get_value(), self.precon_rank
            )
            self.preconditioner_matrix.requires_grad_(True)  # Ensure it tracks gradients

        else:
            raise NotImplementedError(f"Preconditioner '{self.precon_type}' is not implemented.")


    def initialize_trackers(self):
        self.Us = []
        self.Rs = []
        self.gammas = []
        self.ps = []
        self.k = -1

    def update_trackers(self, x0, r0, gamma0, p0, k):
        if self.track_iterations:
            self.Us.append(x0.clone().cpu())
            self.Rs.append(r0.clone().cpu())
            self.gammas.append(gamma0.clone().cpu())
            self.ps.append(p0.clone().cpu())
            self.k = k

    def fit(self, X, y):
        self.X_train = X.to(dtype=self.dtype, device=self.device)
        y = y.to(dtype=self.dtype, device=self.device)
        n = self.X_train.shape[0]

        # Precompute the kernel matrix once and cache it
        # with torch.no_grad():

        self.get_preconditioner(self.X_train)
        self.K_lazy = self.kernel(self.X_train, self.X_train)
        def matmul_closure(b):
            return self.K_lazy.matmul(b) + self.noise.get_value()**2 * b


        #-------- Setup RHS of mbCG -------- 
        b = torch.empty((n, self.num_probes + 1), dtype=self.dtype, device=self.device)
        b[:, 0] = y.view(-1)

        if self.num_probes > 0:
            if self.precon_type == "piv_chol":
                k = self.L_k.shape[1]
                # Sample z1 ~ NORMAL(0, I_k) and z2 ~ NORMAL(0, I_n)
                z1 = torch.randn((k, self.num_probes), dtype=self.dtype, device=self.device)
                z2 = torch.randn((n, self.num_probes), dtype=self.dtype, device=self.device)
                sigma = self.noise.get_value()
                # Form random probe vectors via reparameterization: L_k @ z1 + sigma * z2
                rand_vectors = self.L_k @ z1 + sigma * z2
            else:
                rand_vectors = torch.randn(n, self.num_probes, dtype=self.dtype, device=self.device)
            
            #---- Normalize to lie on the unit sphere ----
            norms = torch.norm(rand_vectors, p=2, dim=0, keepdim=True)
            rand_vectors = rand_vectors / (norms + 1e-10)
            b[:, 1:] = rand_vectors

        # Normalize each column of b to unit norm
        rhs_norm = torch.norm(b, p=2, dim=0, keepdim=True)
        rhs_is_zero = rhs_norm.lt(1e-10)
        rhs_norm = rhs_norm.masked_fill(rhs_is_zero, 1)
        b_normalized = b / rhs_norm

        self.Z = b[:, 1:]
        #---- Warm start initialization ----
        if self.warm_start and self.previous_solutions is not None:
            x0 = self.previous_solutions
        else:
            x0 = None  # This triggers a zeros initial guess inside initialize_cg

        inv_quad_term, logdet_term = InvQuadLogdet(
                    representation_tree,            # how to build your main linear operator from *matrix_args
                    representation_tree,            # or None/precond_representation_tree if you have a real preconditioner
                    identity_precon,                # or None if you do not want a preconditioner closure
                    0,                              # number of arguments used by the preconditioner tree
                                               # we want inv_quad = y^T K^{-1} y
                    rand_vectors,                           # probe_vectors=None => draw random probes inside the function
                    norms,                           # probe_vector_norms=None => the function will compute them
                    inv_quad_rhs,                   # first entry in *args is the RHS
                    *matrix_args                    # the rest of the arguments for building the matrix
                )


       
        solution = x0 * rhs_norm
        self.previous_solution = solution  # Cache for warm start
        self.alpha = solution[:, 0]
        self.probe_solutions = solution[:, 1:]


    def predict(self, X):
        X = X.to(dtype=self.dtype, device=self.device)
        K_trans = self.kernel(self.X_train, X)
        predictive_mean = K_trans.t() @ self.alpha  
        
        if self.compute_covariance:
            # If F has not been computed yet, compute it using Lanczos decomposition.
            if self.F is None:
                # if self_K_train == None:
                K_train = self.kernel(self.X_train, self.X_train).evaluate()
                K_train = K_train+ self.noise.get_value()**2
                K_train = K_train.detach()
                Q, T = lanczos_decomp(K_train, self.pred_lanczos_rank, self.device, self.dtype)
                L = psd_safe_cholesky(T)
                F_t = torch.linalg.solve_triangular(L, Q.t(), upper=False)
                self.F = F_t.t()
            K_test_full = self.kernel(X, X).evaluate()
            beta = self.F.t() @ K_trans
            covariance = K_test_full - beta.t() @ beta
            variance = torch.diag(covariance)
            return predictive_mean, covariance#, variance
        else:
            return predictive_mean,0
