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
from mbcg import linear_cg, initialize_cg,cond_fn,take_cg_step,print_analysis,build_lanczos,lanczos_decomp,build_lanczos_batched
from preconditioners import build_cholesky,identity_precon
import torch
import torch.nn as nn
import gpytorch
from abc import ABC, abstractmethod
import math 

import torch
from gpytorch.lazy import lazify
from gpytorch.functions import pivoted_cholesky


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
        K = self.kernel(X, X).evaluate()
        if self.compute_covariance:
            mean = K_trans @ self.alpha
            v = torch.cholesky_solve(K_trans.t(), self.L)
            covariance = K - K_trans @ v
            return mean.detach().squeeze(-1), covariance.detach().squeeze(-1)
        else:
            return mean.detach().squeeze(-1)

    def compute_mll(self, y): #Exact MLL computation
        n = y.shape[0]
        log_det_K = 2 * torch.sum(torch.log(torch.diagonal(self.L)))
        quadratic_term = torch.dot(y.squeeze(), self.alpha.squeeze())
        const =n * torch.log(torch.tensor(2 * torch.pi, dtype=self.dtype, device=self.device))
        mll = 0.5 * (const + log_det_K + quadratic_term)
       # print("quadratic term",quadratic_term,"log det ",log_det_K,"constant",const)

        return mll


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
            self.preconditioner_matrix = torch.eye(n, dtype=self.dtype, device=self.device)
            self.preconditioner_matrix.requires_grad_(True)  # Ensure it tracks gradients   
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

        if self.K_train ==None:
            self.K_train = self.kernel(self.X_train, self.X_train).evaluate()
            self.K_train.requires_grad_(True)  # Ensure it tracks gradients

        def matmul_closure(b):
            return self.K_train @ b + self.noise.get_value()**2 * b


        #-------- Setup RHS of mbCG -------- 
        b = torch.empty((n, self.num_probes + 1), dtype=self.dtype, device=self.device)
        b[:, 0] = y.view(-1)

        if self.num_probes > 0:
            if self.precon_type == "piv_chol":
                print(self.L_k.shape)
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
            x0 = torch.zeros_like(b_normalized)

        if self.track_iterations and not hasattr(self, 'Us'):
            self.initialize_trackers()

        #---- Build preconditioner/state ----
        state, aux = initialize_cg(
            matmul_closure, b_normalized, stop_updating_after=1e-10,
            eps=1e-10, preconditioner=self.pinv_closure
        )
        x0, has_converged, r0, batch_shape, residual_norm = state
        (p0, gamma0, mul_storage, beta, alpha, is_zero, z0) = aux
        if self.verbose:
            self.update_trackers(x0, r0, gamma0, p0, k=0)

        #-------- CG iteration --------
        alpha_history_list = []
        beta_history_list = []
        iter_idx = 0
    
        for k in range(1, self.cg_max_iter):
            Ap0 = matmul_closure(p0)
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

      # Stack the accumulated rows to create history tensors.
        alpha_history = torch.stack(alpha_history_list, dim=0)
        beta_history = torch.stack(beta_history_list, dim=0)

        # Transpose the history tensors so the probe dimension comes first.
        # Now alpha_batch has shape (num_probes, iter_idx) and beta_history (num_probes, iter_idx).
        alpha_batch = alpha_history.transpose(0, 1)  # shape: (num_probes, iter_idx)
        beta_batch = beta_history.transpose(0, 1)    # shape: (num_probes, iter_idx)
        # For each probe, beta should be of length (iter_idx - 1) (since beta_list[0] corresponds to iteration 2 onward).
        beta_batch = beta_batch[:, :alpha_batch.size(1) - 1]  # shape: (num_probes, iter_idx - 1)
        # Build a batch of Lanczos matrices.
        T_batch = build_lanczos_batched(alpha_batch, beta_batch, dtype=self.dtype, device=self.device, eps=1e-16)
        # T_batch now has shape (num_probes, iter_idx, iter_idx)
        # If needed, convert the batch into a list of individual matrices.
        self.lanczos_iterates = [T_batch[i] for i in range(T_batch.size(0))]

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


    # def compute_mll(self, y):
    #     #https://arxiv.org/pdf/2107.00243

    #     y = y.to(dtype=self.dtype, device=self.device)
    #     n = y.shape[0]

    #     #---- Lancoz Quadrature ----
    #     tau_P_log = torch.logdet(self.preconditioner_matrix)
    #     T_stack = torch.stack(self.lanczos_iterates, dim=0)
    #     eigenvals, eigenvecs = torch.linalg.eigh(T_stack)
    #     weights = eigenvecs[:, 0, :] ** 2
    #     gamma = torch.sum(weights * torch.log(eigenvals), dim=1)
    #     gamma_sum = torch.sum(gamma)

    #     tau_star_log = tau_P_log + (n / self.num_probes) * gamma_sum
    #     quadratic = y.T @ self.alpha
    #     const_term = torch.log(torch.tensor(2 * torch.pi, dtype=self.dtype, device=self.device))
    #     mll = 0.5 * (quadratic + tau_star_log + n * const_term)
    #     return mll     

    def compute_mll(self, y):
        """
        Compute model marginal log-likelihood using Lanczos quadrature.
        Based on https://arxiv.org/pdf/2107.00243
        """
        y = y.to(dtype=self.dtype, device=self.device)
        n = y.shape[0]
        
        # Lanczos Quadrature
        tau_P_log = torch.logdet(self.preconditioner_matrix)
        
        # Stack Lanczos iterates and compute eigendecomposition
        T_stack = torch.stack(self.lanczos_iterates, dim=0)
        with torch.no_grad():
            eigenvals, eigenvecs = torch.linalg.eigh(T_stack)
        
        # Ensure positive eigenvalues for numerical stability
        eigenvals = torch.clamp(eigenvals, min=1e-10)
        
        # Compute quadrature weights and log determinant estimate
        weights = eigenvecs[:, 0, :] ** 2
        gamma = torch.sum(weights * torch.log(eigenvals), dim=1)
        gamma_sum = torch.sum(gamma)
        
        # Total log determinant estimate
        tau_star_log = tau_P_log + (n / self.num_probes) * gamma_sum
        
        # Quadratic term
        quadratic = y.T @ self.alpha
        
        # Constant term
        const_term = n * torch.log(torch.tensor(2 * torch.pi, dtype=self.dtype, device=self.device))
        
        # Final MLL computation
        mll = 0.5 * (quadratic + tau_star_log + const_term)
        #print("quadratic term",quadratic,"log det approx",tau_star_log,"constant",const_term)
        return mll

    # def estimate_mll_gradient(self,train_x):
    #     kernel = self.kernel
    #     noise_param = self.noise.u
    #     n = self.X_train.shape[0]
    #     u0 = self.alpha.view(-1, 1)
    #     U = self.probe_solutions
    #     Z = self.Z
    #     P_inv = torch.linalg.inv(self.preconditioner_matrix)
        
    #     u0_flat = u0.reshape(-1)
    #     u0_outer = torch.outer(u0_flat, u0_flat)
    #     P_inv_Z = torch.matmul(P_inv, Z)
    #     t = Z.shape[1]
        
    #     outer_products = torch.zeros((t, n, n), device=self.X_train.device)
    #     for i in range(t):
    #         outer_products[i] = torch.outer(P_inv_Z[:, i], U[:, i])
        
    #     summed_outer = outer_products.sum(dim=0) / t
    #     for param in kernel.parameters():
    #         param.requires_grad_(True)
        
    #     K = self.kernel(self.K_train,self.K_train).evaluate()+ self.noise.get_value()**2 *torch.eye(self.X_train.shape[0],device=self.device)
        
    #     param_grads = {}
    #     params_list = list(filter(lambda p: p.requires_grad, kernel.parameters()))
    #     param_names = [name for name, param in kernel.named_parameters() if param.requires_grad]
    #     grads_term1 = torch.autograd.grad(
    #         K, params_list, grad_outputs=u0_outer, retain_graph=True, create_graph=False
    #     )
        
    #     grads_term2 = torch.autograd.grad(
    #         K, params_list, grad_outputs=summed_outer, retain_graph=True, create_graph=False
    #     )
        
    #     for i, (param_name, grad1, grad2) in enumerate(zip(param_names, grads_term1, grads_term2)):
    #         param_grads[param_name] = 0.5 * (grad1 + grad2)
        
    #     trace_u0 = torch.trace(u0_outer)
    #     trace_summed = torch.trace(summed_outer)

    #     noise_value = self.noise.get_value() 
    #     noise_u = self.noise.u
    #     noise_grad_estimate = noise_value * torch.sigmoid(noise_u) * (trace_u0 + trace_summed)
    #     param_grads["noise"] = noise_grad_estimate
        
    #     return param_grads
    def estimate_mll_gradient(self, train_x):
        """
        Estimate gradients of the MLL with respect to kernel parameters using
        stochastic Lanczos quadrature.
        """
        kernel = self.kernel
        n = self.X_train.shape[0]
        
        # Get solution vectors
        u0 = self.alpha.view(-1, 1)
        U = self.probe_solutions
        Z = self.Z
        
        # Compute preconditioner inverse
        P_inv = torch.linalg.inv(self.preconditioner_matrix)
        
        # Compute outer product for the first term
        u0_flat = u0.reshape(-1)
        u0_outer = torch.outer(u0_flat, u0_flat)
        
        # Compute outer products for the second term
        P_inv_Z = torch.matmul(P_inv, Z)
        t = Z.shape[1]
        
        # Initialize outer products tensor
        outer_products = torch.zeros((t, n, n), device=self.device, dtype=self.dtype)
        
        # Compute individual outer products
        for i in range(t):
            outer_products[i] = torch.outer(P_inv_Z[:, i], U[:, i])
        
        # Average the outer products
        summed_outer = outer_products.sum(dim=0) / t
        
        # Enable gradient tracking for kernel parameters
        for param in kernel.parameters():
            param.requires_grad_(True)
        
        # Compute kernel matrix with noise diagonal
        K = self.K_train + self.noise.get_value()**2 * torch.eye(n, device=self.device, dtype=self.dtype)
        
        # Initialize parameter gradients dictionary
        param_grads = {}
        
        # Get list of parameters that require gradients
        params_list = list(filter(lambda p: p.requires_grad, kernel.parameters()))
        param_names = [name for name, param in kernel.named_parameters() if param.requires_grad]
        
        # Compute gradients for the first term
        grads_term1 = torch.autograd.grad(
            outputs=K, 
            inputs=params_list, 
            grad_outputs=u0_outer, 
            retain_graph=True, 
            create_graph=False
        )
        
        # Compute gradients for the second term
        grads_term2 = torch.autograd.grad(
            outputs=K, 
            inputs=params_list, 
            grad_outputs=summed_outer, 
            retain_graph=True, 
            create_graph=False
        )
        
        # Combine gradients
        for i, (param_name, grad1, grad2) in enumerate(zip(param_names, grads_term1, grads_term2)):
            param_grads[param_name] = 0.5 * (grad1 - grad2)  # Changed sign for second term
        
        # Compute noise gradient
        trace_u0 = torch.trace(u0_outer)
        trace_summed = torch.trace(summed_outer)
        noise_value = self.noise.get_value()
        noise_u = self.noise.u
        
        # Calculate noise gradient with proper sigmoid derivative
        sigmoid_derivative = torch.sigmoid(noise_u) * (1 - torch.sigmoid(noise_u))
        noise_grad_estimate = noise_value * sigmoid_derivative * (trace_u0 - trace_summed)  # Changed sign
        
        param_grads["noise"] = noise_grad_estimate
        
        return param_grads

    def apply_mll_gradient(model, lr):
        # Compute the gradient estimates for the hyperparameters.
        param_grads = model.estimate_mll_gradient()

        # Update each kernel parameter using gradient ascent.
        for name, param in model.kernel.named_parameters():
            if param.requires_grad and name in param_grads:
                param.data = param.data - lr * param_grads[name]

        # Update the noise parameter consistently with gradient ascent.
        if "noise" in param_grads:
            model.noise.u.data = model.noise.u.data - lr * param_grads["noise"]




