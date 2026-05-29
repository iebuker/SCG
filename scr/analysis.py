import math

import numpy as np

from .utils import to_numpy

__all__ = [
    "cov2cor",
    "rv_coefficient",
    "simulate_data_cov",
    "pick_kernel_params",
    "classify_edges",
]


# -------------------------------------------------------------------
# Helper metrics: cov2cor, RV, summaries
# -------------------------------------------------------------------


def cov2cor(Sigma):
    """Convert a covariance matrix to the corresponding correlation matrix.

    Parameters
    ----------
    Sigma : array_like, shape (p, p)
        A covariance matrix.

    Returns
    -------
    numpy.ndarray, shape (p, p)
        A correlation matrix.
    """
    Sigma = np.asarray(Sigma, dtype=float)
    d = np.sqrt(np.diag(Sigma))
    d[d <= 0] = np.inf
    R = Sigma.copy()
    R /= d[np.newaxis, :]
    R /= d[:, np.newaxis]
    R = np.clip(R, -1.0, 1.0)
    return R


def rv_coefficient(A, B):
    """Compute RV coefficient between two matrices.

    Parameters
    ----------
    A, B : array_like, shape (p, p)
        Symmetric matrices.

    Returns
    -------
    float
        RV coefficient in ``[0, 1]``; ``numpy.nan`` if either matrix is zero.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    num = np.sum(A * B)
    denom = math.sqrt(np.sum(A * A) * np.sum(B * B))
    if denom <= 0:
        return np.nan
    return num / denom


# -------------------------------------------------------------------
# Kernel parameter tuning
# -------------------------------------------------------------------


def pick_kernel_params(
    init,
    S,
    score_quantile=0.9,
    min_score=0.1,
    alpha_clip=(0.1, 50.0),
    eps=1e-12,
):
    """Heuristic for kernel hyperparameters from an init.

    Given init, this function estimates a length-scale ``rho``
    and ``alpha`` for RQ kernel used in the  GP prior covariance ``Kmat``.
    For every feature (i.e. genes) pair it forms the init implied spatial
    covariance field, estimates a length-scale by regressing
    ``-log|cov|`` on squared distance through the origin, keeps the pairs with
    the strongest signal (i.e. largest mean correlation magnitude), and averages
    their length-scales. ``alpha`` is derived from the dispersion (i.e. coefficient
    of variation) of the per-pair inverse squared length-scales: low
    dispersion implies a near-squared-exponential kernel (large ``alpha``),
    high dispersion implies a heavier-tailed kernel (small ``alpha``).

    Parameters
    ----------
    init : dict
        An initialisation dict as returned by :func:`scr.model.scr_init`
        (uses keys ``mu_Theta``, ``mu_Xi``, ``a_sigma``, ``b_sigma``).
    S : array_like, shape (n, d)
        Spatial coordinates of the ``n`` locations.
    score_quantile : float, optional
        Quantile of the per-pair correlation-magnitude score above which a
        pair is retained for length-scale averaging. Defaults to ``0.9``.
    min_score : float, optional
        Floor on the score threshold. Defaults to ``0.1``.
    alpha_clip : tuple of float, optional
        ``(low, high)`` clip range for the returned ``alpha``. Defaults to
        ``(0.1, 50.0)``.
    eps : float, optional
        Small constant guarding logarithms and divisions. Defaults to
        ``1e-12``.

    Returns
    -------
    dict
        ``{"alpha": float, "rho": float}`` -- suggested kernel hyperparameters
        (consumable by :func:`scr.utils.rq_kernel`). Falls back to
        ``{"alpha": 1.0, "rho": 1.0}`` when no usable signal is found.

    Raises
    ------
    ValueError
        If the shapes of ``S`` and the arrays in ``init`` are inconsistent.
    """

    mu_Theta_t = init["mu_Theta"]  # (p, L) torch
    mu_Xi_t = init["mu_Xi"]  # (N, L, K) torch
    a_sigma_t = init["a_sigma"]  # (p,) torch
    b_sigma_t = init["b_sigma"]  # (p,) torch

    mu_Theta = mu_Theta_t.detach().cpu().numpy()
    mu_Xi = mu_Xi_t.detach().cpu().numpy()
    a_sigma = a_sigma_t.detach().cpu().numpy()
    b_sigma = b_sigma_t.detach().cpu().numpy()

    # sigma^2 estimate: 1 / E[precision] = 1 / (a_sigma / b_sigma)
    sigma2_hat = 1.0 / (a_sigma / b_sigma)  # (p,)

    S = np.asarray(S, dtype=float)
    N, d = S.shape

    N_xi, L, K = mu_Xi.shape
    p, L_theta = mu_Theta.shape

    if N_xi != N:
        raise ValueError(
            f"pick_kernel_params: nrow(S)={N} must equal dim(mu_Xi)[0]={N_xi}"
        )
    if L_theta != L:
        raise ValueError(
            f"pick_kernel_params: mu_Theta.shape[1]={L_theta} "
            f"must equal mu_Xi.shape[1]={L}"
        )

    # ---------------------------------------------------------
    # Build Sigma_arr: p x p x N covariance field
    #    Sigma_i = (Theta * Xi_i) (Theta * Xi_i)^T + diag(sigma2_hat)
    # ---------------------------------------------------------
    Sigma_arr = np.empty((p, p, N), dtype=float)

    for i in range(N):
        Xi_i = mu_Xi[i, :, :]  # (L, K)
        Li = mu_Theta @ Xi_i  # (p, K)
        Sigma_i = Li @ Li.T  # (p, p)
        Sigma_i += np.diag(sigma2_hat.astype(float))
        Sigma_arr[:, :, i] = Sigma_i

    # ---------------------------------------------------------
    # Convert to correlation matrices per spot
    # ---------------------------------------------------------
    # Correlation fields are only used for scoring the signal strength of the
    Cor_arr = np.empty_like(Sigma_arr)
    for i in range(N):
        Cor_arr[:, :, i] = cov2cor(Sigma_arr[:, :, i])

    # ---------------------------------------------------------
    # Helper: estimate lengthscale from a covariance field v
    #    v has length N and corresponds to cov at each spatial location.
    # ---------------------------------------------------------
    def rho_from_field(v):
        v = np.asarray(v, dtype=float)
        idx = np.argmax(np.abs(v))
        v0 = abs(v[idx])
        if v0 < eps:
            return np.inf  # essentially flat field

        # squared distances from "center" location idx
        diff = S - S[idx, :]  # (N, d)
        r2 = np.sum(diff**2, axis=1)  # (N,)

        # y = -log(|v|/v0)
        ratio = np.abs(v) / v0
        ratio = np.maximum(ratio, eps)
        y = -np.log(ratio)

        # regression y ~ r2 through origin: slope = sum(r2*y) / sum(r2^2)
        num = np.sum(r2 * y)
        den = np.sum(r2 * r2)
        if den <= 0.0 or num <= 0.0:
            return np.inf

        k = num / den
        rho_val = 1.0 / np.sqrt(2.0 * k)
        return rho_val

    # ---------------------------------------------------------
    # Loop over gene–gene pairs (upper triangle)
    # ---------------------------------------------------------
    i_idx, j_idx = np.triu_indices(p, k=1)
    n_pairs = i_idx.size

    rho_v = np.empty(n_pairs, dtype=float)
    score = np.empty(n_pairs, dtype=float)

    for pp in range(n_pairs):
        g1 = i_idx[pp]
        g2 = j_idx[pp]

        v_cov = Sigma_arr[g1, g2, :]  # (N,)
        v_cor = Cor_arr[g1, g2, :]  # (N,)

        rho_v[pp] = rho_from_field(v_cov)
        score[pp] = np.nanmean(np.abs(v_cor))

    # ---------------------------------------------------------
    # Filter by score and compute rho_hat, alpha_hat
    # ---------------------------------------------------------

    finite_mask = np.isfinite(rho_v)
    if not np.any(finite_mask):
        # total fallback
        return {"alpha": 1.0, "rho": 1.0}

    # score threshold
    score_finite = score[np.isfinite(score)]
    if score_finite.size == 0:
        sc_thresh = min_score
    else:
        q_val = np.nanquantile(score_finite, score_quantile)
        sc_thresh = max(min_score, q_val)

    sub_mask = finite_mask & (score >= sc_thresh)

    if not np.any(sub_mask):
        # fallback: use all finite rho
        sub_mask = finite_mask

    if not np.any(sub_mask):
        # total fallback
        return {"alpha": 1.0, "rho": 1.0}

    rho_vec = rho_v[sub_mask]
    rho_hat = float(np.mean(rho_vec))

    # alpha from dispersion of w = 1 / rho^2
    w = 1.0 / (rho_vec**2)
    m_w = np.mean(w)
    if rho_vec.size > 1:
        sd_w = np.std(w, ddof=1)
    else:
        sd_w = 0.0

    if m_w <= 0.0 or not np.isfinite(m_w):
        cv_w = np.nan
    else:
        cv_w = sd_w / m_w

    if (not np.isfinite(cv_w)) or (cv_w <= 0.0):
        alpha_hat = 50.0  # nearly SE
    else:
        alpha_hat = 1.0 / (cv_w**2)

    # clip alpha to prevent too large/small values
    alpha_low, alpha_high = alpha_clip
    alpha_hat = max(alpha_low, min(alpha_hat, alpha_high))

    return {"alpha": float(alpha_hat), "rho": rho_hat}


def simulate_data_cov(
    N,
    p,
    K_spatial=3,
    K_const=1,
    frac_genes_spatial=0.3,
    frac_genes_const=0.7,
    noise_var=0.05,
    grid_eps=1e-8,
    seed=None,
    save_rng_state=True,
):
    """Simulate a spatial covariance-regression dataset with known SCG pairs.

    Draws each expression vector from a location-specific multivariate Gaussian,
    ``Y_i ~ N(0, Sigma_i)``, with ``Sigma_i = Lambda_i Lambda_i^T +
    noise_var * I`` and ``Lambda_i = Theta * B(s_i)``. A subset of genes load
    on spatially varying factor fields ``B(s)`` while the remainder load
    on a constant factor.

    Locations lie on a regular square grid (see :func:`simulate_data`), so the
    realised count is ``N_eff = ceil(sqrt(N))**2``.

    Parameters
    ----------
    N : int
        Target number of locations; the grid yields ``N_eff >= N``.
    p : int
        Number of features (genes).
    K_spatial : int, optional
        Number of spatially varying factors. Defaults to ``3``.
    K_const : int, optional
        Number of constant (non-spatial) factors. Defaults to ``1``.
    frac_genes_spatial : float, optional
        Fraction of genes loading on spatial factors. Defaults to ``0.3``.
    frac_genes_const : float, optional
        Fraction of genes loading on the constant factor. Defaults to ``0.7``.
    noise_var : float, optional
        Diagonal observation-noise variance added to every ``Sigma_i``.
        Defaults to ``0.05``.
    grid_eps : float, optional
        Small inset keeping grid points inside ``[-1, 1]``. Defaults to
        ``1e-8``.
    seed : int or None, optional
        Seed for the random generator. A random seed is drawn when ``None``.
    save_rng_state : bool, optional
        If ``True`` (default) include the generator's bit-generator state in
        the returned ``info`` for exact reproducibility.

    Returns
    -------
    Y : numpy.ndarray, shape (N_eff, p)
        Simulated observations.
    S : numpy.ndarray, shape (N_eff, 2)
        Location coordinates.
    info : dict
        Ground truth and metadata: ``seed``; ``rng_state``; ``scg_pairs`` and
        ``non_scg_pairs`` (arrays of gene-index pairs); ``noise_var``; and
        ``Sigma_array`` (N_eff, p, p), the true covariance field.
    """
    # --------------------------------------------------
    # Seed handling
    # --------------------------------------------------
    if seed is None:
        seed = int(np.random.SeedSequence().entropy)
    rng = np.random.default_rng(seed)

    # --------------------------------------------------
    # Spatial grid
    # --------------------------------------------------
    num_grids = int(np.ceil(np.sqrt(N)))
    grid_vals = np.linspace(-1.0 + grid_eps, 1.0 - grid_eps, num_grids)
    xv, yv = np.meshgrid(grid_vals, grid_vals, indexing="xy")
    S = np.column_stack([xv.ravel(), yv.ravel()])
    N_eff = S.shape[0]
    x, y = S[:, 0], S[:, 1]

    # --------------------------------------------------
    # Basis functions
    # --------------------------------------------------
    def f1(x, y):
        return x

    def f2(x, y):
        return y

    def f3(x, y):
        return x + y

    def f4(x, y):
        return x - y

    def f5(x, y):
        return 0.5 * x + 0.2 * y

    def f6(x, y):
        return 0.2 * x - 0.5 * y

    def f7(x, y):
        return x + 0.1

    def f8(x, y):
        return y - 0.1

    def f9(x, y):
        return 0.7 * x + 0.3 * y

    def f10(x, y):
        return 0.3 * x - 0.7 * y

    def f11(x, y):
        return x**2

    def f12(x, y):
        return y**2

    def f13(x, y):
        return x**2 + y**2

    def f14(x, y):
        return x**2 - y**2

    def f15(x, y):
        return x * y

    def f16(x, y):
        return (x + y) ** 2

    def f17(x, y):
        return (x - y) ** 2

    def f18(x, y):
        return x**2 + 0.5 * y

    def f19(x, y):
        return y**2 - 0.5 * x

    def f20(x, y):
        return x**2 + 0.5 * y**2

    def f21(x, y):
        return x**3

    def f22(x, y):
        return y**3

    def f23(x, y):
        return x**3 + y**3

    def f24(x, y):
        return x**3 - y**3

    def f25(x, y):
        return x**4 + y**2

    def f26(x, y):
        return x**2 + y**4

    def f27(x, y):
        return x**4 + y**4

    def f28(x, y):
        return x**2 * y

    def f29(x, y):
        return x * y**2

    def f30(x, y):
        return x**2 * y**2

    def f31(x, y):
        return (x + 0.3 * y) ** 2

    def f32(x, y):
        return (y - 0.2 * x) ** 2

    def f33(x, y):
        return (x + 0.2) ** 2 + (y - 0.2) ** 2

    def f34(x, y):
        return (x - 0.4) ** 2 + 0.5 * (y + 0.3) ** 2

    def f35(x, y):
        return 0.5 * (x + y) ** 2 + 0.2 * (x - y)

    def f36(x, y):
        return 0.6 * x**2 + 0.4 * y**2 + 0.2 * x * y

    def f37(x, y):
        return 0.4 * x**2 + 0.6 * y**2 - 0.2 * x * y

    def f38(x, y):
        return (x**2 + y**2) ** 2

    def f39(x, y):
        return (x**2 + y**2) ** 3

    def f40(x, y):
        return 0.3 * (x**3 + y**3) + 0.4 * (x + y)

    def f41(x, y):
        return np.exp(-(x**2 + y**2))

    def f42(x, y):
        return np.exp(-2 * (x**2 + y**2))

    def f43(x, y):
        return np.exp(-3 * (x**2 + y**2))

    def f44(x, y):
        return np.exp(-(x**2 + 2 * y**2))

    def f45(x, y):
        return np.exp(-(2 * x**2 + y**2))

    def f46(x, y):
        return np.exp(-((x - 0.3) ** 2 + (y + 0.2) ** 2) / 0.15)

    def f47(x, y):
        return np.exp(-((x + 0.4) ** 2 + (y - 0.4) ** 2) / 0.2)

    def f48(x, y):
        return np.exp(-((x + 0.1) ** 2) / 0.05) * np.exp(-(y**2) / 0.2)

    def f49(x, y):
        return np.exp(-0.8 * (x**2 + y**2)) * (x + y)

    def f50(x, y):
        return (x - y) * np.exp(-0.8 * (x**2 + y**2))

    def f51(x, y):
        return 1.0 / (1.0 + x**2 + y**2)

    def f52(x, y):
        return 1.0 / (1.0 + 2 * x**2 + 0.5 * y**2)

    def f53(x, y):
        return 1.0 / np.sqrt(1.0 + x**2 + y**2)

    def f54(x, y):
        return (x + y) / (1.0 + x**2 + y**2)

    def f55(x, y):
        return (x - y) / (1.0 + x**2 + y**2)

    def f56(x, y):
        return (x**2 + y**2) / (1.0 + x**2 + y**2)

    def f57(x, y):
        return (x**2 - y**2) / (1.0 + x**2 + y**2)

    def f58(x, y):
        return (x + 0.3 * y) / (1.0 + 0.5 * x**2 + 0.2 * y**2)

    def f59(x, y):
        return (0.5 * x - 0.2 * y) / (1.0 + 0.2 * x**2 + 0.8 * y**2)

    def f60(x, y):
        return (x**2 + 0.3 * y**2) / (1.0 + 0.5 * (x**2 + y**2))

    def f61(x, y):
        return np.tanh(x)

    def f62(x, y):
        return np.tanh(y)

    def f63(x, y):
        return np.tanh(x + y)

    def f64(x, y):
        return np.tanh(2 * x - y)

    def f65(x, y):
        return 0.5 * np.tanh(2 * x) + 0.5 * np.tanh(2 * y)

    def f66(x, y):
        return np.tanh(1.5 * y)

    def f67(x, y):
        return 1.0 / (1.0 + np.exp(-3 * (x + 0.5 * y)))

    def f68(x, y):
        return 1.0 / (1.0 + np.exp(-4 * (x - y)))

    def f69(x, y):
        return 0.5 + 0.5 * np.tanh(2 * (x + 0.3 * y))

    def f70(x, y):
        return 0.5 + 0.5 * np.tanh(2 * (y - 0.3 * x))

    def f71(x, y):
        return np.log1p(np.exp(2 * x)) / 2.0

    def f72(x, y):
        return np.log1p(np.exp(2 * y)) / 2.0

    def f73(x, y):
        return np.log1p(np.exp(3 * (x + y))) / 3.0

    def f74(x, y):
        return np.log1p(np.exp(3 * (x - y))) / 3.0

    def f75(x, y):
        return (np.log1p(np.exp(4 * (x + 0.2))) + np.log1p(np.exp(4 * (y - 0.2)))) / 4.0

    def f76(x, y):
        return np.log1p(np.exp(3 * (0.6 * x - 0.4 * y))) / 3.0

    def f77(x, y):
        return (np.log1p(np.exp(2 * x)) - np.log1p(np.exp(-2 * x))) / 4.0

    def f78(x, y):
        return (np.log1p(np.exp(2 * y)) - np.log1p(np.exp(-2 * y))) / 4.0

    def f79(x, y):
        return (np.log1p(np.exp(2 * (x + y))) - np.log1p(np.exp(-2 * (x + y)))) / 4.0

    def f80(x, y):
        return (np.log1p(np.exp(2 * (x - y))) - np.log1p(np.exp(-2 * (x - y)))) / 4.0

    def f81(x, y):
        return np.exp(0.5 * x) - 1.0

    def f82(x, y):
        return np.exp(0.5 * y) - 1.0

    def f83(x, y):
        return np.exp(0.4 * (x + y)) - 1.0

    def f84(x, y):
        return np.exp(0.4 * (x - y)) - 1.0

    def f85(x, y):
        return np.exp(-0.5 * (x**2 + y**2))

    def f86(x, y):
        return (x + y) * np.exp(-0.5 * (x**2 + y**2))

    def f87(x, y):
        return (x - y) * np.exp(-0.5 * (x**2 + y**2))

    def f88(x, y):
        return (x**2 + y**2) * np.exp(-0.5 * (x**2 + y**2))

    def f89(x, y):
        return (x**2 - y**2) * np.exp(-0.5 * (x**2 + y**2))

    def f90(x, y):
        return (0.3 * x + 0.7 * y) * np.exp(-0.8 * (x**2 + y**2))

    def f91(x, y):
        return (x + y + 0.5 * (x**2 + y**2)) * np.exp(-0.5 * (x**2 + y**2))

    def f92(x, y):
        return 0.4 * x + 0.6 * y + 0.2 * x * y

    def f93(x, y):
        return 0.2 + 0.8 * np.exp(-(x**2 + y**2))

    def f94(x, y):
        return 0.3 + 0.7 / (1.0 + (x + 0.4 * y) ** 2)

    def f95(x, y):
        return 0.5 + 0.5 * np.tanh(2 * (x + 0.3 * y))

    def f96(x, y):
        return (x + y) ** 3 / 6.0

    def f97(x, y):
        return (x - y) ** 3 / 6.0

    def f98(x, y):
        return (x + 0.5 * y) ** 2

    def f99(x, y):
        return (0.5 * x - y) ** 2

    def f100(x, y):
        return ((x**2 + y**2) + 0.5 * (x + y)) * np.exp(-(x**2 + y**2))

    basis_fns = [
        f1,
        f2,
        f3,
        f4,
        f5,
        f6,
        f7,
        f8,
        f9,
        f10,
        f11,
        f12,
        f13,
        f14,
        f15,
        f16,
        f17,
        f18,
        f19,
        f20,
        f21,
        f22,
        f23,
        f24,
        f25,
        f26,
        f27,
        f28,
        f29,
        f30,
        f31,
        f32,
        f33,
        f34,
        f35,
        f36,
        f37,
        f38,
        f39,
        f40,
        f41,
        f42,
        f43,
        f44,
        f45,
        f46,
        f47,
        f48,
        f49,
        f50,
        f51,
        f52,
        f53,
        f54,
        f55,
        f56,
        f57,
        f58,
        f59,
        f60,
        f61,
        f62,
        f63,
        f64,
        f65,
        f66,
        f67,
        f68,
        f69,
        f70,
        f71,
        f72,
        f73,
        f74,
        f75,
        f76,
        f77,
        f78,
        f79,
        f80,
        f81,
        f82,
        f83,
        f84,
        f85,
        f86,
        f87,
        f88,
        f89,
        f90,
        f91,
        f92,
        f93,
        f94,
        f95,
        f96,
        f97,
        f98,
        f99,
        f100,
    ]
    # n_basis = len(basis_fns)

    # --------------------------------------------------
    # Spatial factor fields B(s)
    # --------------------------------------------------
    basis_idx = rng.choice(len(basis_fns), size=K_spatial, replace=False)

    B_spatial = np.zeros((N_eff, K_spatial))
    for j, idx in enumerate(basis_idx):
        vals = basis_fns[idx](x, y)
        vals = (vals - vals.mean()) / (vals.std() + 1e-8)
        B_spatial[:, j] = vals

    B_const = rng.normal(size=(1, K_const))
    B_const = np.repeat(B_const, N_eff, axis=0)

    B_all = np.concatenate([B_spatial, B_const], axis=1)
    # K = K_spatial + K_const

    # --------------------------------------------------
    # Gene loadings
    # --------------------------------------------------
    n_genes_spatial = int(round(frac_genes_spatial * p))
    n_genes_const = min(int(round(frac_genes_const * p)), p - n_genes_spatial)

    perm = rng.permutation(p)
    genes_spatial = perm[:n_genes_spatial]
    genes_const = perm[n_genes_spatial : n_genes_spatial + n_genes_const]

    Theta_spatial = np.zeros((p, K_spatial))
    Theta_const = np.zeros((p, K_const))

    for g in genes_spatial:
        k = rng.integers(1, K_spatial + 1)
        idx = rng.choice(K_spatial, size=k, replace=False)
        Theta_spatial[g, idx] = rng.normal(0, 1, size=k)

    for g in genes_const:
        k = rng.integers(1, K_const + 1)
        idx = rng.choice(K_const, size=k, replace=False)
        Theta_const[g, idx] = rng.normal(0, 0.1, size=k)

    Theta_all = np.concatenate([Theta_spatial, Theta_const], axis=1)

    # --------------------------------------------------
    # Simulate Y
    # --------------------------------------------------
    Y = np.empty((N_eff, p))
    Sigma = np.zeros((N_eff, p, p))
    for i in range(N_eff):
        Lambda_i = Theta_all * B_all[i]  # (p, K)
        Sigma_i = Lambda_i @ Lambda_i.T  # (p, p)
        Sigma_i += noise_var * np.eye(p)  # diagonal noise

        Sigma[i] = Sigma_i
        Y[i] = rng.multivariate_normal(mean=np.zeros(p), cov=Sigma_i)

    # --------------------------------------------------
    # Info object
    # --------------------------------------------------
    info = {
        "seed": seed,
        "rng_state": rng.bit_generator.state if save_rng_state else None,
        "noise_var": noise_var,
        "Sigma_array": Sigma,
    }

    return Y, S, info


# -------------------------------------------------------------------
# Edge classification (Otsu + Bayesian FDR)
# -------------------------------------------------------------------


def _otsu_threshold(x):
    """Otsu threshold via histogram inter-class variance.

    Returns the value ``t`` that maximises
    ``w0(t) * w1(t) * (mu0(t) - mu1(t))**2``, with class weights and means
    taken from a 256-bin histogram of ``x`` (matching scikit-image's default).
    When the optimum forms a plateau (i.e. the histogram has
    an empty gap between two well-separated modes) the threshold is set at
    the midpoint of the plateau, as opposed to the miniumum used by
    scikit-image's default.
    """
    x = np.asarray(x, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    lo, hi = float(x.min()), float(x.max())
    if lo == hi:
        return lo
    nbins = 256
    edges = np.linspace(lo, hi, nbins + 1)
    counts, _ = np.histogram(x, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    p = counts.astype(float) / counts.sum()
    w0 = np.cumsum(p)
    mu_cum = np.cumsum(p * centers)
    mu_total = mu_cum[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        # sigma_b^2 = (mu_T * w0 - mu_cum)^2 / (w0 * (1 - w0))
        sigma_b = (mu_total * w0 - mu_cum) ** 2 / (w0 * (1.0 - w0))
    sigma_b[~np.isfinite(sigma_b)] = -np.inf
    # mid-plateau when ties occur
    max_val = sigma_b.max()
    tied = np.flatnonzero(sigma_b == max_val)
    return float(centers[(tied[0] + tied[-1]) // 2])


def classify_edges(
    params, M=500, alpha=0.1, seed=None, gene_names=None, tie_break_by_sd=True
):
    """Classify gene-gene edges as spatially co-varying by Bayesian FDR.

    Given a fitted variational posterior (the ``params`` dict returned by
    :func:`scr.cavi`), this routine draws ``M`` independent samples of
    ``Theta``, ``Xi`` and ``sigma^2``; for each sample and each canonical
    edge ``(g, g')`` it forms the implied gene-gene correlation across the
    ``n`` spatial locations and takes its spatial standard deviation; an
    Otsu threshold ``t_otsu`` on the posterior-mean per-edge spatial SDs
    separates a "flat" group from a "varying" group; the per-edge posterior
    error probability is

        PEP[e] = P( spatial_SD[e] <= t_otsu | data ),

    estimated as the fraction of posterior samples whose per-sample spatial
    SD falls below ``t_otsu``. Edges are then selected by Bayesian FDR
    control at level ``alpha``: sort by ascending PEP (and, optionally, by
    descending posterior-mean SD as a tie-break), then keep the largest
    set whose cumulative-mean PEP stays at or below ``alpha``.

    Every unordered gene pair appears in the returned frame exactly once,
    with ``gene1 < gene2`` lexicographically when ``gene_names`` are strings
    (or numerically otherwise), so the same edge is never represented twice.

    Memory
    ------
    Posterior samples are drawn one at a time and only the per-edge spatial
    SDs are retained (i.e. an ``M`` by ``E`` array), avoiding the
    ``(M, p, K, n)`` Lambda-samples array a naive implementation would
    materialise.

    Parameters
    ----------
    params : dict
        The ``params`` dict returned by :func:`scr.cavi`. Required keys:
        ``mu_Theta``, ``Sigma_Theta_list``, ``mu_Xi``, ``var_Xi``,
        ``a_sigma``, ``b_sigma``. Torch tensors are accepted; conversion to
        NumPy happens internally.
    M : int, optional
        Number of posterior samples drawn. Defaults to ``1000``.
    alpha : float, optional
        Bayesian-FDR control level. Defaults to ``0.1``.
    seed : int or None, optional
        Seed for NumPy's random generator. Defaults to ``None`` (no reseed).
    gene_names : sequence or None, optional
        Length-``p`` labels for the features. Defaults to integer indices
        ``0, 1, ..., p-1``.
    tie_break_by_sd : bool, optional
        If ``True`` (default), break PEP ties by descending posterior-mean
        spatial SD -- preferring edges with stronger spatial variation.

    Returns
    -------
    pandas.DataFrame
        One row per canonical edge, with columns ``edge`` (``"gene1-gene2"``),
        ``gene1``, ``gene2``, ``q_value`` (= ``1 - PEP``: the posterior
        probability that the edge *is* spatially varying), and ``scg``
        (selected at the requested FDR level). The frame's ``.attrs``
        dictionary also carries ``otsu_threshold``, ``alpha``, ``M`` and
        ``n_selected``.
    """
    import pandas as pd

    mu_Theta = to_numpy(params["mu_Theta"])  # (p, L)
    mu_Xi = to_numpy(params["mu_Xi"])  # (n, L, K)
    var_Xi = to_numpy(params["var_Xi"])  # (n, L, K)
    a_sigma = to_numpy(params["a_sigma"]).ravel()  # (p,)
    b_sigma = to_numpy(params["b_sigma"]).ravel()  # (p,)
    Sigma_Theta_list = params["Sigma_Theta_list"]

    p_n, L = mu_Theta.shape
    n, _, K = mu_Xi.shape

    # Marginal posterior variances of Theta: diagonals of Sigma_Theta_g
    var_Theta = np.empty_like(mu_Theta)
    for g in range(p_n):
        var_Theta[g] = np.diag(to_numpy(Sigma_Theta_list[g]))

    if gene_names is None:
        gene_names = np.arange(p_n)
    else:
        gene_names = np.asarray(gene_names)
        if gene_names.size != p_n:
            raise ValueError(
                f"gene_names must have length p={p_n} (got {gene_names.size})"
            )

    g_idx, gp_idx = np.triu_indices(p_n, k=1)
    E = g_idx.size

    name_g = gene_names[g_idx]
    name_gp = gene_names[gp_idx]
    # If labels are strings/objects, swap each pair so name_g < name_gp.
    # For numeric labels np.triu_indices already produces g_idx < gp_idx.
    if gene_names.dtype.kind in ("U", "S", "O"):
        swap = name_g > name_gp
        if np.any(swap):
            tmp = name_g[swap].copy()
            name_g[swap] = name_gp[swap]
            name_gp[swap] = tmp

    # Stream posterior samples; record per-sample spatial SDs
    if seed is not None:
        np.random.seed(int(seed))

    sd_Theta = np.sqrt(var_Theta)
    sd_Xi = np.sqrt(var_Xi)
    sd_per_sample = np.empty((int(M), E), dtype=float)

    for s in range(int(M)):
        Theta_s = mu_Theta + np.random.standard_normal(mu_Theta.shape) * sd_Theta
        Xi_s = mu_Xi + np.random.standard_normal(mu_Xi.shape) * sd_Xi
        prec_s = np.random.gamma(shape=a_sigma, scale=1.0 / b_sigma)
        sigma2_s = 1.0 / prec_s  # (p,)

        # Lambda_s[i, g, k] = sum_l Theta_s[g, l] * Xi_s[i, l, k]
        Lambda_s = np.einsum("gl,ilk->igk", Theta_s, Xi_s)  # (n, p, K)
        LLT = np.einsum("igk,ihk->igh", Lambda_s, Lambda_s)  # (n, p, p)
        V = np.diagonal(LLT, axis1=1, axis2=2) + sigma2_s[None, :]  # (n, p)

        cov_e = LLT[:, g_idx, gp_idx]  # (n, E)
        denom_e = np.sqrt(V[:, g_idx] * V[:, gp_idx])  # (n, E)
        corr_e = cov_e / denom_e  # (n, E)
        sd_per_sample[s] = corr_e.std(axis=0, ddof=1)  # (E,)

    # Posterior-mean spatial SD, Otsu, PEP
    sd_mean = sd_per_sample.mean(axis=0)
    t_otsu = _otsu_threshold(sd_mean)
    pep = (sd_per_sample <= t_otsu).mean(axis=0)  # (E,)

    # Bayesian-FDR selection
    if tie_break_by_sd:
        order = np.lexsort((-sd_mean, pep))  # primary asc pep, then desc sd
    else:
        order = np.argsort(pep, kind="mergesort")
    pep_sorted = pep[order]
    cume_mean = np.cumsum(pep_sorted) / (np.arange(E) + 1)
    ok = np.where(cume_mean <= alpha)[0]
    selected = np.zeros(E, dtype=bool)
    if ok.size > 0:
        kmax = int(ok.max() + 1)
        selected[order[:kmax]] = True

    edge_str = np.array([f"{a}-{b}" for a, b in zip(name_g, name_gp)])
    df = pd.DataFrame(
        {
            "edge": edge_str,
            "gene1": name_g,
            "gene2": name_gp,
            "q_value": 1.0 - pep,
            "scg": selected,
        }
    )
    df.attrs["otsu_threshold"] = float(t_otsu)
    df.attrs["alpha"] = float(alpha)
    df.attrs["M"] = int(M)
    df.attrs["n_selected"] = int(selected.sum())
    return df
