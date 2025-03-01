from chex import Array, PRNGKey
from functools import partial
from typing import Callable
from structs import KernelParams, FeatureParams
import jax
import jax.numpy as jnp
import jax.random as jr

from typing import NamedTuple, Optional
from chex import Array, PRNGKey


class Dataset(NamedTuple):
    x: Array
    y: Array
    N: int
    D: int

    mu_x: Optional[Array] = None
    sigma_x: Optional[Array] = None
    mu_y: Optional[Array] = None
    sigma_y: Optional[Array] = None


class KernelParams(NamedTuple):
    signal_scale: float
    length_scale: Array


class FeatureParams(NamedTuple):
    omega: Array
    phi: Array


class ModelParams(NamedTuple):
    noise_scale: float
    kernel_params: KernelParams


class TrainState(NamedTuple):
    key: PRNGKey
    v0: Array
    z: Array
    feature_params: FeatureParams
    w: Array
    eps: Array



def _sq_dist(x1: Array, x2: Array, length_scale: Array):
    x1, x2 = x1 / length_scale[None, :], x2 / length_scale[None, :]
    sq_dist = jnp.sum((x1[:, None] - x2[None, :]) ** 2, axis=-1)
    sq_dist = jnp.clip(sq_dist, a_min=1e-10, a_max=None)
    return sq_dist


def _kernel_grad_fn(kernel_fn: Callable):
    def _f(x1, x2, kernel_params):
        return jnp.squeeze(kernel_fn(x1, x2, kernel_params))

    return jax.vmap(
        jax.vmap(jax.grad(_f, argnums=2), in_axes=(None, 0, None)),
        in_axes=(0, None, None),
    )


def _phi_fn(key: PRNGKey, n_features: int):
    return jr.uniform(key=key, shape=(1, n_features), minval=-jnp.pi, maxval=jnp.pi)


def _feature_params_fn_sin_cos(
    key: PRNGKey, n_input_dims: int, n_features: int, omega_fn: Callable
):
    assert n_features % 2 == 0
    omega = omega_fn(key, n_input_dims, int(n_features / 2))
    return FeatureParams(omega=omega, phi=None)


def _feature_params_fn_cos(
    key: PRNGKey, n_input_dims: int, n_features: int, omega_fn: Callable
):
    omega_key, phi_key = jr.split(key, 2)
    omega = omega_fn(omega_key, n_input_dims, n_features)
    phi = _phi_fn(phi_key, n_features)
    return FeatureParams(omega=omega, phi=phi)


def _feature_fn_sin_cos(x: Array, kernel_params: KernelParams, feature_params: FeatureParams):
    n_features = 2 * feature_params.omega.shape[1]
    norm = jnp.sqrt(2.0 / n_features)
    arg = (x / kernel_params.length_scale[None, :]) @ feature_params.omega
    feat = jnp.concatenate([jnp.sin(arg), jnp.cos(arg)], axis=-1)
    return kernel_params.signal_scale * norm * feat


def _feature_fn_cos(x: Array, kernel_params: KernelParams, feature_params: FeatureParams):
    n_features = feature_params.omega.shape[1]
    norm = jnp.sqrt(2.0 / n_features)
    feat = jnp.cos(
        (x / kernel_params.length_scale[None, :]) @ feature_params.omega
        + feature_params.phi
    )
    return kernel_params.signal_scale * norm * feat


def feature_fn(x: Array, kernel_params: KernelParams, feature_params: FeatureParams):
    return _feature_fn_sin_cos(x, kernel_params, feature_params)


def feature_vec_prod(x: Array, kernel_params: KernelParams, feature_params: FeatureParams, vec: Array, batch_size: int = 1):
    n, d = x.shape
    padding = (batch_size - (n % batch_size)) % batch_size
    x = jnp.concatenate([x, jnp.zeros((padding, d))], axis=0)

    def _f(carry, idx):
        return carry, feature_fn(x[idx], kernel_params, feature_params) @ vec

    xs = jnp.reshape(jnp.arange(x.shape[0]), (-1, batch_size))
    _, prod = jax.lax.scan(_f, None, xs)
    return jnp.squeeze(jnp.reshape(prod, (x.shape[0], -1)))[:n]


def rbf_kernel_fn(x1: Array, x2: Array, kernel_params: KernelParams):
    return (kernel_params.signal_scale**2) * jnp.exp(
        -0.5 * _sq_dist(x1, x2, kernel_params.length_scale)
    )


def rbf_kernel_grad_fn(x1: Array, x2: Array, kernel_params: KernelParams):
    return _kernel_grad_fn(rbf_kernel_fn)(x1, x2, kernel_params)


def _rbf_omega_fn(key: PRNGKey, n_input_dims: int, n_features: int):
    return jr.normal(key, shape=(n_input_dims, n_features))


def rbf_feature_params_fn(key: PRNGKey, n_input_dims: int, n_features: int):
    return _feature_params_fn_sin_cos(key, n_input_dims, n_features, _rbf_omega_fn)


_matern12_df, _matern32_df, _matern52_df = 1.0, 3.0, 5.0


def _matern_kernel_fn(
    x1: Array, x2: Array, kernel_params: KernelParams, df: float, normaliser: Callable
):
    sq_dist = _sq_dist(x1, x2, kernel_params.length_scale)
    dist = jnp.sqrt(sq_dist)

    normaliser = normaliser(dist, sq_dist)
    exponential_term = jnp.exp(-jnp.sqrt(df) * dist)
    return (kernel_params.signal_scale**2) * normaliser * exponential_term


def _matern_omega_fn(key: PRNGKey, n_input_dims: int, n_features: int, df: float):
    return jr.t(key, df=df, shape=(n_input_dims, n_features))


def matern12_feature_params_fn(key: PRNGKey, n_input_dims: int, n_features: int):
    return _feature_params_fn_sin_cos(
        key, n_input_dims, n_features, partial(_matern_omega_fn, df=_matern12_df)
    )


def matern32_feature_params_fn(key: PRNGKey, n_input_dims: int, n_features: int):
    return _feature_params_fn_sin_cos(
        key, n_input_dims, n_features, partial(_matern_omega_fn, df=_matern32_df)
    )


def matern52_feature_params_fn(key: PRNGKey, n_input_dims: int, n_features: int):
    return _feature_params_fn_sin_cos(
        key, n_input_dims, n_features, partial(_matern_omega_fn, df=_matern52_df)
    )


def _matern12_normaliser(dist: Array, sq_dist: Array):
    return 1.0


def _matern32_normaliser(dist: Array, sq_dist: Array):
    return jnp.sqrt(3.0) * dist + 1.0


def _matern52_normaliser(dist: Array, sq_dist: Array):
    return jnp.sqrt(5.0) * dist + (5.0 / 3.0) * sq_dist + 1.0


def matern12_kernel_fn(x1: Array, x2: Array, kernel_params: KernelParams):
    return _matern_kernel_fn(
        x1, x2, kernel_params, df=_matern12_df, normaliser=_matern12_normaliser
    )


def matern32_kernel_fn(x1: Array, x2: Array, kernel_params: KernelParams):
    return _matern_kernel_fn(
        x1, x2, kernel_params, df=_matern32_df, normaliser=_matern32_normaliser
    )


def matern52_kernel_fn(x1: Array, x2: Array, kernel_params: KernelParams):
    return _matern_kernel_fn(
        x1, x2, kernel_params, df=_matern52_df, normaliser=_matern52_normaliser
    )


def matern12_kernel_grad_fn(x1: Array, x2: Array, kernel_params: KernelParams):
    return _kernel_grad_fn(matern12_kernel_fn)(x1, x2, kernel_params)


def matern32_kernel_grad_fn(x1: Array, x2: Array, kernel_params: KernelParams):
    return _kernel_grad_fn(matern32_kernel_fn)(x1, x2, kernel_params)


def matern52_kernel_grad_fn(x1: Array, x2: Array, kernel_params: KernelParams):
    return _kernel_grad_fn(matern52_kernel_fn)(x1, x2, kernel_params)
