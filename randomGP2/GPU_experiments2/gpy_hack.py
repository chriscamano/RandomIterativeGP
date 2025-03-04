
import math
import warnings
from typing import Callable, Optional, Tuple

import torch
import gpytorch
from torch import Tensor

from linear_operator.operators.added_diag_linear_operator import AddedDiagLinearOperator
from linear_operator import settings
from linear_operator.utils.warnings import NumericalWarning

class CustomPrecondtionedGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel,preconditioner_closure=None):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.preconditioner_closure = None
        self.covar_module = preconditioner_closure

    def forward(self, x,dump_args=False):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        if dump_args:
            return CustomMultivariateNormal(mean_x, covar_x,self.preconditioner_closure),mean_x,covar_x #<-- subject,
        return CustomMultivariateNormal(mean_x, covar_x,self.preconditioner_closure) #<-- subject

def custom_preconditioner(operator: AddedDiagLinearOperator) -> Tuple[Optional[Callable], Optional[torch.Tensor], Optional[torch.Tensor]]:
    print("DEBUG: USING CUSTOM INJECTION in custom_preconditioner1 ")
    if settings.max_preconditioner_size.value() == 0 or operator.size(-1) < settings.min_preconditioning_size.value():
        return None, None, None
    if operator._q_cache is None:
        max_iter = settings.max_preconditioner_size.value()
        operator._piv_chol_self = operator._linear_op.pivoted_cholesky(rank=max_iter)
        if torch.any(torch.isnan(operator._piv_chol_self)).item():
            warnings.warn("NaNs encountered in custom preconditioner computation. Attempting to continue without preconditioning.", NumericalWarning)
            return None, None, None
        operator._init_cache()
    def precondition_closure(tensor: torch.Tensor) -> torch.Tensor:
        qqt = operator._q_cache.matmul(operator._q_cache.mT.matmul(tensor))
        if operator._constant_diag:
            return (1 / operator._noise) * (tensor - qqt)
        else:
            return (tensor / operator._noise) - qqt
    return precondition_closure, operator._precond_lt, operator._precond_logdet_cache

from linear_operator.operators.dense_linear_operator import DenseLinearOperator
from linear_operator.operators.dense_linear_operator import DenseLinearOperator

class CustomDenseLinearOperator(DenseLinearOperator):
    def __init__(self, tensor: torch.Tensor, preconditioner_closure=None):
        super().__init__(tensor)
        self._preconditioner_closure = preconditioner_closure

    def _preconditioner(self):
        if self._preconditioner_closure is not None:
            return self._preconditioner_closure(self)
        return super()._preconditioner()

class CustomMultivariateNormal(gpytorch.distributions.MultivariateNormal):
    def __init__(self, mean: Tensor, covariance_matrix, custom_preconditioner=None):
        if hasattr(covariance_matrix, "evaluate_kernel"):
            evaluated_cov = covariance_matrix.evaluate_kernel()
        else:
            evaluated_cov = covariance_matrix
        from linear_operator.operators.dense_linear_operator import DenseLinearOperator
        if isinstance(evaluated_cov, torch.Tensor):
            evaluated_cov = DenseLinearOperator(evaluated_cov)
        if isinstance(evaluated_cov, DenseLinearOperator) and not isinstance(evaluated_cov, CustomDenseLinearOperator):
            evaluated_cov = CustomDenseLinearOperator(evaluated_cov.tensor, preconditioner_closure=custom_preconditioner)
        if not hasattr(evaluated_cov, "_preconditioner_closure"):
            print("DEBUG: Evaluated covariance is of type", type(evaluated_cov), "which does not support our custom preconditioner storage.")
        super().__init__(mean, evaluated_cov)
        self.custom_preconditioner = custom_preconditioner

    def log_prob(self, value: Tensor) -> Tensor:
        if gpytorch.settings.fast_computations.log_prob.off():
            return super().log_prob(value)
        if self._validate_args:
            self._validate_sample(value)
        mean, covar = self.loc, self.lazy_covariance_matrix
        diff = value - mean
        if diff.shape[:-1] != covar.batch_shape:
            if len(diff.shape[:-1]) < len(covar.batch_shape):
                diff = diff.expand(covar.shape[:-1])
            else:
                padded_batch_shape = (*(1 for _ in range(diff.dim() + 1 - covar.dim())), *covar.batch_shape)
                diff = diff.expand(padded_batch_shape)
                covar = covar.repeat(*(diff_size // covar_size for diff_size, covar_size in zip(diff.shape[:-1], padded_batch_shape)), 1, 1)
        #print("before eval", type(covar))
        evaluated_cov = covar.evaluate_kernel()
        evaluated_cov = CustomDenseLinearOperator(evaluated_cov.tensor, preconditioner_closure=self.custom_preconditioner)
        #print("after eval", type(evaluated_cov))
        #print()
        inv_quad, logdet = evaluated_cov.inv_quad_logdet(inv_quad_rhs=diff.unsqueeze(-1), logdet=True)
        res = -0.5 * (inv_quad + logdet + diff.size(-1) * math.log(2 * math.pi))
        return res

class CustomExactMarginalLogLikelihood(gpytorch.mlls.ExactMarginalLogLikelihood):
    def __init__(self, likelihood, model, custom_preconditioner=None):
        super().__init__(likelihood, model)
        self.custom_preconditioner = custom_preconditioner

    def forward(self, function_dist, target, *params):
        transformed_dist = self.likelihood(function_dist, *params)
        if not isinstance(transformed_dist, CustomMultivariateNormal):
            #print("DEBUG: Wrapping likelihood-transformed function_dist in CustomMultivariateNormal.")
            transformed_dist = CustomMultivariateNormal(transformed_dist.mean, transformed_dist.covariance_matrix, custom_preconditioner=self.custom_preconditioner)
        res = transformed_dist.log_prob(target)
        res = self._add_other_terms(res, params)
        num_data = transformed_dist.event_shape.numel()
        return res.div_(num_data)

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x: Tensor, train_y: Tensor, likelihood: gpytorch.likelihoods.GaussianLikelihood, kernel: gpytorch.kernels.Kernel):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel
    def dump_args(self,x):
        return self.mean_module(x), self.covar_module(x)
    def forward(self, x: Tensor,dump_args=False):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)