import random

import numpy as np
import torch

# ==========================================================================
# Global defaults
# ==========================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""torch.device: default compute device -- CUDA when available, else CPU."""

DTYPE = torch.float64
"""torch.dtype: default float dtype. Double precision stabilises CAVI solves."""

__all__ = [
    "DEVICE",
    "DTYPE",
    "rademacher_vec",
    "solve_spd",
    "rq_kernel",
    "se_kernel",
    "frob",
    "frob_array",
    "l2",
    "set_seed",
    "to_numpy",
    "safe_div",
    "sample_theta",
    "sample_xi",
    "sample_sigma2",
    "canonical_edge",
]


# ==========================================================================
# Basic helpers
# ==========================================================================


def rademacher_vec(n, device=DEVICE, dtype=DTYPE):
    """
    Rademacher vectors are the probe vectors behind the Hutchinson estimators
    used in this package: for any matrix ``A`` and a vector ``z`` with i.i.d.
    +/-1 entries, ``E[z * (A z)] = diag(A)`` and ``E[z^T A z] = trace(A)``.
    See :func:`scr.model.update_xi`, which uses this identity to estimate the
    marginal posterior variances of the GP coefficients.

    Parameters
    ----------
    n : int
        Length of the vector.
    device : torch.device, optional
        Device on which to allocate the result. Defaults to :data:`DEVICE`.
    dtype : torch.dtype, optional
        Floating-point dtype of the result. Defaults to :data:`DTYPE`.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(n,)`` with entries in ``{-1, +1}``.
    """
    z = torch.randint(0, 2, (n,), device=device)
    return (2 * z - 1).to(dtype)


def frob(A):
    """Frobenius norm of a tensor.

    Parameters
    ----------
    A : torch.Tensor
        Input tensor of any shape.

    Returns
    -------
    torch.Tensor
        Scalar tensor ``sqrt(sum(A**2))``.
    """
    return torch.linalg.norm(A)


def frob_array(A, B):
    """Frobenius norm of the difference ``A - B``.

    Used by :func:`scr.model.cavi` to measure the change in a parameter block
    between successive CAVI iterations.

    Parameters
    ----------
    A, B : torch.Tensor
        Tensors of identical shape.

    Returns
    -------
    torch.Tensor
        Scalar tensor ``||A - B||_F``.
    """
    return torch.linalg.norm(A - B)


def l2(x):
    """L2 norm of an array.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor; treated as a flat vector.

    Returns
    -------
    torch.Tensor
        Scalar tensor ``||x||_2``.
    """
    return torch.linalg.norm(x)


def set_seed(seed, deterministic=False):
    """Seed the Python, NumPy and PyTorch random number generators.

    Parameters
    ----------
    seed : int
        Seed value shared by all three RNGs (and the CUDA RNGs if present).
    deterministic : bool, optional
        If ``True``, additionally request deterministic algorithms from
        PyTorch/cuDNN. This can slow computation and may raise if a used op
        has no deterministic implementation. Defaults to ``False``.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ==========================================================================
# Robust SPD solver (low-allocation)
# ==========================================================================


def solve_spd(A, B, jitter=1e-8, max_tries=3, symmetrize=False):
    """Solve the linear system ``A X = B`` for an SPD matrix ``A``.

    Parameters
    ----------
    A : torch.Tensor
        Square coefficient matrix of shape ``(d, d)``.
    B : torch.Tensor
        Right-hand side of shape ``(d,)`` or ``(d, m)``. Passing the identity
        yields ``X = A^{-1}``.
    jitter : float, optional
        Diagonal jitter added on the first failed attempt. Subsequent failures
        multiply it by 10. Defaults to ``1e-8``.
    max_tries : int, optional
        Maximum number of solve attempts. Defaults to ``3``.
    symmetrize : bool, optional
        If ``True``, replace ``A`` with ``(A + A^T) / 2`` before solving.
        Defaults to ``False`` (callers usually symmetrize beforehand).

    Returns
    -------
    X : torch.Tensor
        Solution, same trailing shape as ``B``.
    A_work : torch.Tensor
        The (possibly jittered) matrix actually factorised. Useful for
        diagnostics; equals ``A`` when no jitter was needed.

    Raises
    ------
    RuntimeError
        If all ``max_tries`` attempts fail.
    """
    if symmetrize:
        A = 0.5 * (A + A.T)

    eps = 0.0
    last_err = None

    for _ in range(max_tries):
        if eps == 0.0:
            A_work = A
        else:
            A_work = A.clone()
            diag = A_work.diagonal()
            diag.add_(eps)

        try:
            X = torch.linalg.solve(A_work, B)
            return X, A_work
        except RuntimeError as e:
            last_err = e
            eps = jitter if eps == 0.0 else eps * 10.0

    raise RuntimeError(f"solve_spd failed after {max_tries} attempts: {last_err}")


# ==========================================================================
# Stationary covariance kernels (low allocation)
# ==========================================================================


def rq_kernel(S, v2=1.0, rho=0.3, alpha=1.0, jitter=1e-5):
    """Rational-quadratic covariance matrix

    Builds the ``n x n`` kernel matrix

        K[i, j] = v2 * (1 + ||S[i] - S[j]||^2 / (2 * alpha * rho^2))^(-alpha)

    plus ``jitter`` on the diagonal for numerical positive-definiteness.

    Parameters
    ----------
    S : array_like, shape (n, d)
        Spatial coordinates of the ``n`` locations.
    v2 : float, optional
        Marginal variance (kernel amplitude). Defaults to ``1.0``.
    rho : float, optional
        Length-scale. Defaults to ``0.3``.
    alpha : float, optional
        Scale-mixture / tail-weight parameter (> 0). Smaller values give
        heavier tails. Defaults to ``1.0``.
    jitter : float, optional
        Constant added to the diagonal. Defaults to ``1e-5``.

    Returns
    -------
    numpy.ndarray, shape (n, n)
        Symmetric positive-definite covariance matrix.
    """
    S = np.asarray(S, dtype=float)
    n, d = S.shape

    sq = np.sum(S**2, axis=1, keepdims=True)
    K = sq + sq.T - 2.0 * (S @ S.T)
    np.maximum(K, 0.0, out=K)

    K /= 2.0 * alpha * rho**2
    K += 1.0
    K[:] = v2 * (K ** (-alpha))

    idx = np.arange(n)
    K[idx, idx] += jitter
    return K


def se_kernel(S, v2=1.0, rho=None, jitter=1e-5):
    """Squared-exponential covariance matrix over locations.

    Builds the ``n x n`` kernel matrix

        K[i, j] = v2 * exp(-||S[i] - S[j]||^2 / (2 * rho^2))

    plus ``jitter`` on the diagonal.

    Parameters
    ----------
    S : array_like, shape (n, d)
        Spatial coordinates of the ``n`` locations.
    v2 : float, optional
        Marginal variance (kernel amplitude). Defaults to ``1.0``.
    rho : float or None, optional
        Length-scale. If ``None`` (default) the median pairwise distance is
        used.
    jitter : float, optional
        Constant added to the diagonal. Defaults to ``1e-5``.

    Returns
    -------
    numpy.ndarray, shape (n, n)
        Symmetric positive-definite covariance matrix.
    """
    S = np.asarray(S, dtype=float)
    n, d = S.shape

    sq = np.sum(S**2, axis=1, keepdims=True)
    K = sq + sq.T - 2.0 * (S @ S.T)
    np.maximum(K, 0.0, out=K)

    if rho is None:
        if n > 1:
            dists = np.sqrt(K[np.triu_indices(n, k=1)])
            rho = np.median(dists) if dists.size > 0 else 1.0
        else:
            rho = 1.0

    K *= -1.0 / (2.0 * rho**2)
    np.exp(K, out=K)
    K *= v2

    idx = np.arange(n)
    K[idx, idx] += jitter
    return K


# ==========================================================================
# Conversion helpers
# ==========================================================================


def to_numpy(x):
    """Recursively convert tensors (and nested lists/tuples) to NumPy arrays.

    Parameters
    ----------
    x : torch.Tensor or list or tuple or object
        A tensor, a (possibly nested) list/tuple of tensors, or any other
        object (returned unchanged).

    Returns
    -------
    numpy.ndarray or object
        ``x`` with every tensor detached, moved to CPU and converted to a
        NumPy array; lists/tuples are stacked into a single array.
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, (list, tuple)):
        return np.array([to_numpy(xx) for xx in x])
    else:
        return x


def safe_div(num, denom):
    """Divide ``num`` by ``denom``, returning NaN for a non-positive divisor.

    Parameters
    ----------
    num : float
        Numerator.
    denom : float
        Denominator.

    Returns
    -------
    float
        ``num / denom`` if ``denom > 0``, otherwise ``numpy.nan``.
    """
    return num / denom if denom > 0 else np.nan


# ==========================================================================
# Posterior Monte-Carlo samplers
# ==========================================================================


def sample_theta(mu_Theta, var_Theta, S):
    """Draw Monte-Carlo samples of ``Theta`` from its variational posterior.

    Treats the entries of ``Theta`` as independent Gaussians with the given
    means and variances.

    Parameters
    ----------
    mu_Theta : numpy.ndarray, shape (p, L)
        Posterior means of ``Theta``.
    var_Theta : numpy.ndarray, shape (p, L)
        Posterior marginal variances of ``Theta``.
    S : int
        Number of Monte-Carlo samples.

    Returns
    -------
    numpy.ndarray, shape (S, p, L)
        Independent posterior draws of ``Theta``.
    """
    eps = np.random.randn(S, *mu_Theta.shape)
    return mu_Theta[None, :, :] + eps * np.sqrt(var_Theta)[None, :, :]


def sample_xi(mu_Xi, var_Xi, S):
    """Draw Monte-Carlo samples of ``Xi`` from its variational posterior.

    Treats the entries of ``Xi`` as independent Gaussians with the given
    means and marginal variances ``var_Xi``.

    Parameters
    ----------
    mu_Xi : numpy.ndarray, shape (n, L, K)
        Posterior means of ``Xi``.
    var_Xi : numpy.ndarray, shape (n, L, K)
        Posterior marginal variances of ``Xi``.
    S : int
        Number of Monte-Carlo samples.

    Returns
    -------
    numpy.ndarray, shape (S, n, L, K)
        Independent posterior draws of ``Xi``.
    """
    eps = np.random.randn(S, *mu_Xi.shape)
    return mu_Xi[None, :, :, :] + eps * np.sqrt(var_Xi)[None, :, :, :]


def sample_sigma2(a_sigma, b_sigma, S):
    """Draw Monte-Carlo samples of the noise variances ``sigma^2``.

    Each ``sigma_g^2`` has an inverse-gamma variational posterior with shape
    ``a_sigma[g]`` and rate ``b_sigma[g]``. Samples are obtained by drawing
    the precision ``1/sigma_g^2 ~ Gamma(a_sigma, rate=b_sigma)`` and
    reciprocating.

    Parameters
    ----------
    a_sigma : numpy.ndarray, shape (p,)
        Posterior shape parameters.
    b_sigma : numpy.ndarray, shape (p,)
        Posterior rate parameters.
    S : int
        Number of Monte-Carlo samples.

    Returns
    -------
    numpy.ndarray, shape (S, p)
        Independent posterior draws of ``sigma^2``.
    """
    prec = np.random.gamma(
        shape=a_sigma[None, :],
        scale=1.0 / b_sigma[None, :],
        size=(S, a_sigma.shape[0]),
    )
    return 1.0 / prec


# ==========================================================================
# Posterior diagnostics
# ==========================================================================


def canonical_edge(a, b):
    """Return an undirected edge as an ordered ``(min, max)`` integer tuple.

    Gives a canonical key for an unordered pair of feature indices,
    so that edge ``(a, b)`` and edge ``(b, a)`` map to the same dictionary
    key when tabulating.

    Parameters
    ----------
    a, b : int
        Feature indices.

    Returns
    -------
    tuple of int
        ``(min(a, b), max(a, b))``.
    """
    return (int(min(a, b)), int(max(a, b)))
