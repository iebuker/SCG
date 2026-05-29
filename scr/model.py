import math

import numpy as np
import torch
from sklearn.decomposition import SparsePCA

from .utils import (
    DEVICE,
    DTYPE,
    frob,
    frob_array,
    l2,
    solve_spd,
)

__all__ = [
    "compute_M",
    "extract_LK_at_i",
    "compute_Gamma_moments",
    "fitted_from_moments",
    "update_eta",
    "update_Theta",
    "update_phi",
    "update_delta",
    "update_sigma",
    "update_xi",
    "cavi",
    "scr_init",
]


# ============================================
# Moments and fitted values
# ============================================


def compute_M(mu_Theta, Sigma_Theta_list, a_sigma, b_sigma):
    """
    Returns the ``L x L`` matrix

        M = E_q[ Theta^T diag(1/sigma^2) Theta ]
          = sum_g E_q[1/sigma_g^2] * E_q[theta_g theta_g^T],

    with ``E[1/sigma_g^2] = a_sigma[g] / b_sigma[g]`` and
    ``E[theta_g theta_g^T] = mu_Theta[g] mu_Theta[g]^T + Sigma_Theta[g]``.

    Parameters
    ----------
    mu_Theta : torch.Tensor, shape (p, L)
        Variational means of the loadings.
    Sigma_Theta_list : list of torch.Tensor
        Length ``p``; ``Sigma_Theta_list[g]`` is the ``(L, L)`` posterior
        covariance of row ``g`` of ``Theta``.
    a_sigma, b_sigma : torch.Tensor, shape (p,)
        Shape and rate of the inverse-gamma posteriors of ``sigma^2``.

    Returns
    -------
    torch.Tensor, shape (L, L)
        The symmetric matrix ``M``.
    """
    p, L = mu_Theta.shape
    assert a_sigma.shape[0] == p
    assert b_sigma.shape[0] == p

    e_sigma_inv = a_sigma / b_sigma  # p

    # mean part: mu_Theta^T diag(e_sigma_inv) mu_Theta
    W = mu_Theta * e_sigma_inv.unsqueeze(1)  # p x L
    M = mu_Theta.T @ W  # L x L

    # variance part: sum_g e_sigma_inv[g] * Sigma_Theta[g] (vectorized)
    Sig_Theta = torch.stack(Sigma_Theta_list, dim=0)  # p x L x L
    M = M + torch.einsum("g,glm->lm", e_sigma_inv, Sig_Theta)

    return M


def extract_LK_at_i(C, i):
    """Return the ``(L, K)`` slice of a per-spot array at spot ``i``.

    Parameters
    ----------
    C : torch.Tensor, shape (n, L, K)
        A per-spot array (e.g. ``mu_Xi`` or ``var_Xi``).
    i : int
        Spot index.

    Returns
    -------
    torch.Tensor, shape (L, K)
        The slice ``C[i]``.
    """
    return C[i, :, :]  # (L, K)


def compute_Gamma_moments(mu_eta, Sigma_eta_list, mu_Xi, var_Xi):
    """
    For each spot define ``Gamma_i = Xi_i @ eta_i`` in ``R^L``. Under the
    mean-field factorization ``Xi`` and ``eta`` are independent, thus

        E[Gamma_i] = mu_Xi_i @ mu_eta_i
        E[Gamma_i Gamma_i^T] = mu_Xi_i M_eta_i mu_Xi_i^T
                               + diag_r( sum_k var_Xi[i,r,k] * M_eta_i[k,k] )

    with ``M_eta_i = Sigma_eta_i + mu_eta_i mu_eta_i^T``.

    Parameters
    ----------
    mu_eta : torch.Tensor, shape (n, K)
        Variational means of the factor scores.
    Sigma_eta_list : list of torch.Tensor
        Length ``n``; ``(K, K)`` posterior covariances of ``eta_i``.
    mu_Xi : torch.Tensor, shape (n, L, K)
        Variational means of the GP coefficients.
    var_Xi : torch.Tensor, shape (n, L, K)
        Variational marginal variances of the GP coefficients.

    Returns
    -------
    E_Gamma : torch.Tensor, shape (n, L)
        ``E[Gamma_i]`` stacked over spots.
    E_GtGamma : torch.Tensor, shape (L, L)
        ``sum_i E[Gamma_i Gamma_i^T]``, explicitly symmetrised.
    """
    device = mu_eta.device

    n, K = mu_eta.shape
    _, L, K2 = mu_Xi.shape
    assert K2 == K

    # Second moment of eta per spot: M_eta[i] = Sigma_eta[i] + mu_eta[i] mu_eta[i]^T
    Sigma_eta = torch.stack(Sigma_eta_list, dim=0)  # (n, K, K)
    M_eta = Sigma_eta + mu_eta.unsqueeze(2) * mu_eta.unsqueeze(1)  # (n, K, K)

    # E[Gamma_i] = mu_Xi_i @ mu_eta_i
    E_Gamma = torch.einsum("ilk,ik->il", mu_Xi, mu_eta)  # (n, L)

    # E[Gamma Gamma^T] summed over spots: sum_i mu_Xi_i M_eta_i mu_Xi_i^T
    XiM = torch.einsum("ilk,ikm->ilm", mu_Xi, M_eta)  # (n, L, K)
    E_GtGamma = torch.einsum("ilm,ipm->lp", XiM, mu_Xi)  # (L, L)

    # diagonal addition: sum_i sum_k var_Xi[i,r,k] * M_eta[i,k,k]
    Mdiag = torch.diagonal(M_eta, dim1=1, dim2=2)  # (n, K)
    diag_corr = (var_Xi * Mdiag.unsqueeze(1)).sum(dim=(0, 2))  # (L,)
    idx = torch.arange(L, device=device)
    E_GtGamma[idx, idx] += diag_corr

    E_GtGamma = 0.5 * (E_GtGamma + E_GtGamma.T)
    return E_Gamma, E_GtGamma


def fitted_from_moments(mu_Theta, mu_eta, mu_Xi):
    """
    Computes ``fitted[i, g] = sum_{r,k} mu_Theta[g,r] mu_Xi[i,r,k]
    mu_eta[i,k]`` -- the model mean ``Theta @ Xi_i @ eta_i`` evaluated at the
    current variational means.

    Parameters
    ----------
    mu_Theta : torch.Tensor, shape (p, L)
        Variational means of the loadings.
    mu_eta : torch.Tensor, shape (n, K)
        Variational means of the factor scores.
    mu_Xi : torch.Tensor, shape (n, L, K)
        Variational means of the GP coefficients.

    Returns
    -------
    torch.Tensor, shape (n, p)
        The fitted mean matrix.
    """
    device = mu_Theta.device
    dtype = mu_Theta.dtype

    n, K = mu_eta.shape
    p, L = mu_Theta.shape
    assert mu_Xi.shape == (n, L, K)

    fitted = torch.zeros(n, p, device=device, dtype=dtype)

    for k in range(K):
        Xi_k = mu_Xi[:, :, k]  # (n, L)
        term = Xi_k @ mu_Theta.T  # (n, p)
        term = term * mu_eta[:, k : k + 1]  # (n, 1) broadcast
        fitted = fitted + term

    return fitted


# ============================================
# Update blocks
# ============================================


def update_eta(
    Y, mu_Theta, Sigma_Theta_list, mu_Xi, var_Xi, a_sigma, b_sigma, prior_prec=None
):
    """

    Replaces ``q(eta_i) = N(mu_eta[i], Sigma_eta[i])`` by its ELBO-optimal
    Gaussian for every spot ``i``. With prior ``eta_i ~ N(0, PriorPrec^{-1})``
    the coordinate update is

        Sigma_eta_i = ( PriorPrec
                        + E[Xi_i^T (Theta^T diag(1/sigma^2) Theta) Xi_i] )^{-1}
        mu_eta_i = Sigma_eta_i @ ( mu_Xi_i^T mu_Theta^T diag(E[1/sigma^2])
                                      Y[i] ).

    Parameters
    ----------
    Y : torch.Tensor, shape (n, p)
        Observations.
    mu_Theta : torch.Tensor, shape (p, L)
        Variational means of the loadings.
    Sigma_Theta_list : list of torch.Tensor
        Length ``p``; ``(L, L)`` posterior covariances of the rows of ``Theta``.
    mu_Xi, var_Xi : torch.Tensor, shape (n, L, K)
        Variational means and marginal variances of the GP coefficients.
    a_sigma, b_sigma : torch.Tensor, shape (p,)
        Inverse-gamma posterior parameters of ``sigma^2``.
    prior_prec : torch.Tensor or None, optional
        ``(K, K)`` prior precision of ``eta_i``. Defaults to the identity,
        i.e. the standard ``N(0, I_K)`` prior.

    Returns
    -------
    mu_eta : torch.Tensor, shape (n, K)
        Updated variational means.
    Sigma_eta_list : list of torch.Tensor
        Updated ``(K, K)`` posterior covariances, one per spot.
    """
    device = Y.device
    dtype = Y.dtype

    n, p = Y.shape
    p2, L = mu_Theta.shape
    assert p2 == p
    _, L2, K = mu_Xi.shape
    assert L2 == L
    assert var_Xi.shape == (n, L, K)
    assert a_sigma.shape[0] == p
    assert b_sigma.shape[0] == p

    e_sig = a_sigma / b_sigma
    M = compute_M(mu_Theta, Sigma_Theta_list, a_sigma, b_sigma)
    diag_M = torch.diagonal(M)

    if prior_prec is None:
        PriorPrec = torch.eye(K, device=device, dtype=dtype)
    else:
        PriorPrec = prior_prec
        assert PriorPrec.shape == (K, K)

    mu_eta = torch.zeros(n, K, device=device, dtype=dtype)
    Sigma_eta_list = [None] * n

    I_K = torch.eye(K, device=device, dtype=dtype)

    for i in range(n):
        mu_Xi_i = extract_LK_at_i(mu_Xi, i)  # (L, K)
        v_lk = extract_LK_at_i(var_Xi, i)  # (L, K)

        S = mu_Xi_i.T @ M @ mu_Xi_i  # (K, K)
        trow = diag_M @ v_lk  # (K,)
        S = S.clone()
        S[torch.arange(K), torch.arange(K)] += trow

        Prec = PriorPrec + S + 1e-8 * I_K
        Prec = 0.5 * (Prec + Prec.T)

        # Sigma_i = Prec^{-1}
        Sigma_i, _ = solve_spd(Prec, I_K, jitter=1e-10, symmetrize=False)

        rhs_L = mu_Theta.T @ (e_sig * Y[i, :])  # (L,)
        mu_i = Sigma_i @ (mu_Xi_i.T @ rhs_L)  # (K,)

        mu_eta[i, :] = mu_i
        Sigma_eta_list[i] = Sigma_i

    return mu_eta, Sigma_eta_list


def update_Theta(
    Y,
    mu_eta,
    Sigma_eta_list,
    mu_Xi,
    var_Xi,
    a_phi,
    b_phi,
    a_delta,
    b_delta,
    a_sigma,
    b_sigma,
):
    """
    Replaces ``q(theta_g) = N(mu_Theta[g], Sigma_Theta[g])`` by its
    ELBO-optimal Gaussian for every feature ``g``. Row ``g`` has the
    multiplicative-gamma prior precision ``D_g = diag(phi[g, :] * tau)`` with
    ``tau_r = prod_{m<=r} delta_m``; the coordinate update is

        Sigma_Theta_g = ( E[D_g]
                          + E[1/sigma_g^2] * sum_i E[Gamma_i Gamma_i^T] )^{-1}
        mu_Theta_g = Sigma_Theta_g @ ( E[1/sigma_g^2]
                                          * sum_i Y[i,g] E[Gamma_i] )

    Parameters
    ----------
    Y : torch.Tensor, shape (n, p)
        Observations.
    mu_eta : torch.Tensor, shape (n, K)
        Variational means of the factor scores.
    Sigma_eta_list : list of torch.Tensor
        Length ``n``; ``(K, K)`` posterior covariances of ``eta_i``.
    mu_Xi, var_Xi : torch.Tensor, shape (n, L, K)
        Variational means and marginal variances of the GP coefficients.
    a_phi, b_phi : torch.Tensor, shape (p, L)
        Gamma posterior parameters of the local precisions ``phi``.
    a_delta, b_delta : torch.Tensor, shape (L,)
        Gamma posterior parameters of the multiplicative-gamma factors.
    a_sigma, b_sigma : torch.Tensor, shape (p,)
        Inverse-gamma posterior parameters of ``sigma^2``.

    Returns
    -------
    mu_Theta : torch.Tensor, shape (p, L)
        Updated variational means.
    Sigma_Theta_list : list of torch.Tensor
        Updated ``(L, L)`` posterior covariances, one per feature.
    """
    device = Y.device
    dtype = Y.dtype

    n, p = Y.shape
    n2, K = mu_eta.shape
    assert n2 == n
    _, L, K2 = mu_Xi.shape
    assert K2 == K

    assert a_phi.shape == b_phi.shape
    p2, L2 = a_phi.shape
    assert p2 == p and L2 == L
    assert a_sigma.shape[0] == p and b_sigma.shape[0] == p

    e_phi = a_phi / b_phi  # (p, L)
    e_delta = a_delta / b_delta  # (L,)
    e_tau = torch.cumprod(e_delta, dim=0)  # (L,)

    E_Gamma, E_GtGamma = compute_Gamma_moments(mu_eta, Sigma_eta_list, mu_Xi, var_Xi)

    mu_Theta = torch.zeros(p, L, device=device, dtype=dtype)
    Sigma_Theta_list = [None] * p

    I_L = torch.eye(L, device=device, dtype=dtype)

    for g in range(p):
        E_Dg = torch.diag(e_phi[g, :] * e_tau)  # (L, L)
        e_sig_inv_g = a_sigma[g] / b_sigma[g]

        Prec = E_Dg + e_sig_inv_g * E_GtGamma + 1e-8 * I_L
        Prec = 0.5 * (Prec + Prec.T)

        # Sigma_g = Prec^{-1}
        Sigma_g, _ = solve_spd(Prec, I_L, jitter=1e-10, symmetrize=False)

        mu_g = e_sig_inv_g * (E_Gamma.T @ Y[:, g])  # (L,)
        mu_g = Sigma_g @ mu_g

        mu_Theta[g, :] = mu_g
        Sigma_Theta_list[g] = Sigma_g

    return mu_Theta, Sigma_Theta_list


def update_phi(mu_Theta, Sigma_Theta_list, a_delta, b_delta, nu):
    """
    Each ``phi[g, r]`` has a ``Gamma(nu/2, nu/2)`` prior and a Gaussian
    likelihood contribution from ``theta[g, r]``, giving the conjugate update

        q(phi[g,r]) = Gamma( (nu + 1) / 2,
                             (nu + E[tau_r] * E[theta[g,r]^2]) / 2 )

    with ``E[theta[g,r]^2] = mu_Theta[g,r]^2 + Sigma_Theta[g][r,r]`` and
    ``E[tau_r] = cumprod(E[delta])[r]``. The shape ``(nu + 1) / 2`` is a
    constant and is rebuilt from ``nu`` on every call.

    Parameters
    ----------
    mu_Theta : torch.Tensor, shape (p, L)
        Variational means of the loadings.
    Sigma_Theta_list : list of torch.Tensor
        Length ``p``; ``(L, L)`` posterior covariances of the rows of ``Theta``.
    a_delta, b_delta : torch.Tensor, shape (L,)
        Gamma posterior parameters of the multiplicative-gamma factors.
    nu : float
        Degrees-of-freedom hyperparameter of the local-shrinkage prior.

    Returns
    -------
    a_phi, b_phi : torch.Tensor, shape (p, L)
        Updated Gamma posterior parameters of ``phi``.
    """
    device = mu_Theta.device
    dtype = mu_Theta.dtype

    p, L = mu_Theta.shape
    a_phi = torch.full((p, L), (nu + 1.0) / 2.0, device=device, dtype=dtype)

    e_tau = torch.cumprod(a_delta / b_delta, dim=0)  # (L,)

    b_phi = torch.zeros(p, L, device=device, dtype=dtype)
    for g in range(p):
        diag_Sg = torch.diagonal(Sigma_Theta_list[g])  # (L,)
        e_theta2 = mu_Theta[g, :] ** 2 + diag_Sg
        b_phi[g, :] = 0.5 * (nu + e_theta2 * e_tau)

    return a_phi, b_phi


def update_delta(mu_Theta, Sigma_Theta_list, a_phi, b_phi, a_delta_prior, b_delta):
    """
    The cumulative column precisions are ``tau_r = prod_{m<=r} delta_m``.

        H[r] = sum_g E[phi[g,r]] * E[theta[g,r]^2],

    the coordinate-optimal variational posterior of ``delta_h`` is
    ``Gamma(a*_h, b*_h)`` with

        a*_h = a_delta_prior[h] + p * (L - h + 1) / 2
        b*_h = 1 + (1/2) * sum_{r>=h} H[r] * prod_{m<=r, m!=h} E[delta_m]

    Parameters
    ----------
    mu_Theta : torch.Tensor, shape (p, L)
        Variational means of the loadings.
    Sigma_Theta_list : list of torch.Tensor
        Length ``p``; ``(L, L)`` posterior covariances of the rows of ``Theta``.
    a_phi, b_phi : torch.Tensor, shape (p, L)
        Gamma posterior parameters of the local precisions ``phi``.
    a_delta_prior : torch.Tensor, shape (L,)
        PRIOR shape parameters of ``delta`` (e.g. ``[a1, a2, ..., a2]``).
        Constant across iterations; see the notes above.
    b_delta : torch.Tensor, shape (L,)
        Current variational rate parameters of ``delta``.

    Returns
    -------
    new_a_delta : torch.Tensor, shape (L,)
        Posterior shape -- constant across iterations.
    new_b_delta : torch.Tensor, shape (L,)
        Posterior rate after one sequential sweep.
    """
    device = mu_Theta.device
    dtype = mu_Theta.dtype

    p, L = mu_Theta.shape
    e_phi = a_phi / b_phi  # (p, L)

    diagS = torch.stack(
        [torch.diagonal(Sigma_Theta_list[g]) for g in range(p)], dim=0
    )  # (p, L)

    E_theta2 = mu_Theta**2 + diagS  # (p, L)
    H = (e_phi * E_theta2).sum(dim=0)  # (L,)

    # Posterior shape: depends only on dimensions, so recompute from the prior.
    counts = torch.arange(L, 0, -1, device=device, dtype=dtype)  # L, L-1, ..., 1
    new_a_delta = a_delta_prior + 0.5 * p * counts

    # Sequential update of the rates.
    new_b_delta = b_delta.clone()
    for h in range(L):
        e_delta = new_a_delta / new_b_delta  # current variational means
        e_excl = e_delta.clone()
        e_excl[h] = 1.0
        # tau_excl[r] = prod_{m<=r, m!=h} E[delta_m]  (= E[tau_r] / E[delta_h])
        tau_excl = torch.cumprod(e_excl, dim=0)
        suffix_h = torch.dot(H[h:], tau_excl[h:])
        new_b_delta[h] = 1.0 + 0.5 * suffix_h

    return new_a_delta, new_b_delta


def update_sigma(
    Y, mu_Theta, Sigma_Theta_list, mu_eta, Sigma_eta_list, mu_Xi, var_Xi, a0, b0
):
    """
    Each ``sigma_g^2`` has an ``Inverse-Gamma(a0, b0)`` prior; the conjugate
    coordinate update is

        q(sigma_g^2) = Inverse-Gamma( a0 + n/2,  b0 + RSS_g / 2 )

    with the expected residual sum of squares

        RSS_g = sum_i E[ (Y[i,g] - theta_g^T Gamma_i)^2 ]
              = sum_i ( Y[i,g]^2
                        - 2 Y[i,g] mu_g^T E[Gamma_i]
                        + tr( E[theta_g theta_g^T] E[Gamma_i Gamma_i^T] ) ),

    Parameters
    ----------
    Y : torch.Tensor, shape (n, p)
        Observations.
    mu_Theta : torch.Tensor, shape (p, L)
        Variational means of the loadings.
    Sigma_Theta_list : list of torch.Tensor
        Length ``p``; ``(L, L)`` posterior covariances of the rows of ``Theta``.
    mu_eta : torch.Tensor, shape (n, K)
        Variational means of the factor scores.
    Sigma_eta_list : list of torch.Tensor
        Length ``n``; ``(K, K)`` posterior covariances of ``eta_i``.
    mu_Xi, var_Xi : torch.Tensor, shape (n, L, K)
        Variational means and marginal variances of the GP coefficients.
    a0, b0 : torch.Tensor
        Inverse-gamma prior shape and rate. Either 1-element tensors
        (broadcast to all ``p`` features) or length-``p`` tensors.

    Returns
    -------
    a_sigma, b_sigma : torch.Tensor, shape (p,)
        Updated inverse-gamma posterior parameters.
    """
    device = Y.device
    dtype = Y.dtype

    n, p = Y.shape
    p2, L = mu_Theta.shape
    assert p2 == p
    n2, K = mu_eta.shape
    assert n2 == n

    if a0.numel() == 1:
        a0 = a0[0] * torch.ones(p, device=device, dtype=dtype)
    if b0.numel() == 1:
        b0 = b0[0] * torch.ones(p, device=device, dtype=dtype)
    assert a0.shape[0] == p and b0.shape[0] == p

    a_sigma = a0 + 0.5 * float(n) * torch.ones(p, device=device, dtype=dtype)

    # Second moment of eta per spot: M_eta[i] = Sigma_eta[i] + mu_eta[i] mu_eta[i]^T
    Sigma_eta = torch.stack(Sigma_eta_list, dim=0)  # (n, K, K)
    M_eta = Sigma_eta + mu_eta.unsqueeze(2) * mu_eta.unsqueeze(1)  # (n, K, K)

    # G_sum = sum_i E[Gamma_i Gamma_i^T]   (Gamma_i = Xi_i eta_i)
    XiM = torch.einsum("ilk,ikm->ilm", mu_Xi, M_eta)  # (n, L, K)
    G_sum = torch.einsum("ilm,ipm->lp", XiM, mu_Xi)  # (L, L), sums over spots
    Mdiag = torch.diagonal(M_eta, dim1=1, dim2=2)  # (n, K)
    diag_corr = (var_Xi * Mdiag.unsqueeze(1)).sum(dim=(0, 2))  # (L,)
    idx = torch.arange(L, device=device)
    G_sum[idx, idx] += diag_corr

    # E[Gamma_i] = Xi_i eta_i
    XiEta = torch.einsum("ilk,ik->il", mu_Xi, mu_eta)  # (n, L)

    # Quadratic term per gene: sum_i tr(E[theta_g theta_g^T] E[Gamma_i Gamma_i^T])
    #   = mu_g^T G_sum mu_g + tr(Sigma_Theta_g G_sum)
    Sig_Theta = torch.stack(Sigma_Theta_list, dim=0)  # (p, L, L)
    quad = (mu_Theta @ G_sum * mu_Theta).sum(dim=1) + (Sig_Theta * G_sum).sum(
        dim=(1, 2)
    )  # (p,)

    # Cross term per gene: sum_i 2 Y[i,g] mu_g^T E[Gamma_i]
    C = Y.T @ XiEta  # (p, L)
    cross = 2.0 * (mu_Theta * C).sum(dim=1)  # (p,)

    ySq = (Y * Y).sum(dim=0)  # (p,)
    rss = ySq - cross + quad  # (p,)

    b_sigma = b0 + 0.5 * rss

    return a_sigma, b_sigma


def update_xi(
    Y,
    Kmat,
    mu_Theta,
    Sigma_Theta_list,
    mu_eta,
    Sigma_eta_list,
    mu_Xi,
    var_Xi,
    a_sigma,
    b_sigma,
    m=30,
    jitter=1e-8,
):
    """
    Each ``(r, k)`` slice ``Xi[:, r, k]`` (i.e. a length-``n`` spatial field with
    GP prior ``N(0, Kmat)``) is updated as a block, conditional on the
    other slices. Writing ``alpha_i`` for the per-spot likelihood precision of
    the slice, the posterior covariance is
    ``S = (Kmat^{-1} + diag(alpha))^{-1}``, evaluated via the Woodbury form

        S = Kmat - Kmat (Kmat + diag(1/alpha))^{-1} Kmat,

    and the posterior mean solves the same system. The marginal variances
    ``diag(S)`` are obtained with a Hutchinson estimator using ``m``
    Rademacher probe vectors (set ``m=0`` for the exact but ``O(n^3)``
    dense diagonal). The mean solve and the Hutchinson solves share a single
    factorisation of the working matrix.

    Parameters
    ----------
    Y : torch.Tensor, shape (n, p)
        Observations.
    Kmat : torch.Tensor, shape (n, n)
        Spatial GP prior covariance over the ``n`` locations.
    mu_Theta : torch.Tensor, shape (p, L)
        Variational means of the loadings.
    Sigma_Theta_list : list of torch.Tensor
        Length ``p``; ``(L, L)`` posterior covariances of the rows of ``Theta``.
    mu_eta : torch.Tensor, shape (n, K)
        Variational means of the factor scores.
    Sigma_eta_list : list of torch.Tensor
        Length ``n``; ``(K, K)`` posterior covariances of ``eta_i``.
    mu_Xi, var_Xi : torch.Tensor, shape (n, L, K)
        Variational means and marginal variances of the GP coefficients;
        modified in place.
    a_sigma, b_sigma : torch.Tensor, shape (p,)
        Inverse-gamma posterior parameters of ``sigma^2``.
    m : int, optional
        Number of Hutchinson probe vectors for the marginal-variance
        estimate. ``0`` selects the exact dense diagonal. Defaults to ``30``.
    jitter : float, optional
        Diagonal jitter added to the working matrix for a stable solve.
        Defaults to ``1e-8``.

    Returns
    -------
    mu_Xi : torch.Tensor, shape (n, L, K)
        Updated variational means (the same tensor as the input).
    var_Xi : torch.Tensor, shape (n, L, K)
        Updated variational marginal variances (the same tensor as the input).
    """
    device = Y.device
    dtype = Y.dtype

    n, p = Y.shape
    p2, L = mu_Theta.shape
    assert p2 == p
    n2, K = mu_eta.shape
    assert n2 == n
    assert mu_Xi.shape == (n, L, K)
    assert var_Xi.shape == (n, L, K)
    assert a_sigma.shape[0] == p and b_sigma.shape[0] == p
    assert Kmat.shape == (n, n)

    # E[1/sigma_g^2]
    e_sig = a_sigma / b_sigma  # (p,)

    # ---- E[theta_{g,r}^2] (p x L) ----
    E_theta2 = torch.empty(p, L, device=device, dtype=dtype)
    for g in range(p):
        d = torch.diagonal(Sigma_Theta_list[g])
        E_theta2[g, :] = mu_Theta[g, :] ** 2 + d

    # ---- E[eta_{i,k}^2] (n x K) ----
    E_eta2 = torch.empty(n, K, device=device, dtype=dtype)
    for i in range(n):
        d = torch.diagonal(Sigma_eta_list[i])
        E_eta2[i, :] = mu_eta[i, :] ** 2 + d

    # Current fitted values & residual
    fitted = fitted_from_moments(mu_Theta, mu_eta, mu_Xi)
    R_base = Y - fitted

    # s_theta2_all[r] = sum_g E[theta_{g,r}^2] * E[1/sigma_g^2]
    s_theta2_all = E_theta2.T @ e_sig  # (L,)

    # Hutchinson setup
    m_eff = max(int(m), 0)
    Z = None
    KZ = None
    if m_eff > 0:
        # Rademacher probe vectors for the Hutchinson diagonal estimator
        Z = (2 * torch.randint(0, 2, (n, m_eff), device=device) - 1).to(dtype)
        KZ = Kmat @ Z  # (n, m_eff)

    # Preallocate working matrix M = Kmat + (stuff on diag)
    # We keep one copy of Kmat as M, and only overwrite its diagonal each iteration.
    M = Kmat.clone()
    base_diag = M.diagonal().clone()  # diag(Kmat) as baseline

    for k in range(K):
        eta_k = mu_eta[:, k]  # (n,)
        Eeta2k = E_eta2[:, k]  # (n,)

        for r in range(L):
            theta_r = mu_Theta[:, r]  # (p,)
            w_g = e_sig * theta_r  # (p,)
            s_theta2 = s_theta2_all[r]

            # denominator for diagonal "noise" term; clamp to avoid overflow/underflow
            denom = Eeta2k * s_theta2
            denom = torch.clamp(denom, min=1e-12)
            invAlpha = 1.0 / denom
            invAlpha = torch.clamp(invAlpha, max=1e12)  # (n,)

            # Build M = Kmat + diag(jitter + invAlpha) without torch.eye
            diag_M = M.diagonal()
            # diag(Kmat) + jitter + invAlpha
            diag_M.copy_(base_diag + jitter + invAlpha)

            # Right-hand side b = eta_k * (R_base @ w_g + correction)
            tmp = R_base @ w_g  # (n,)
            col = mu_Xi[:, r, k].clone()  # (n,) the current slice being updated
            tmp = tmp + (col * eta_k) * torch.dot(theta_r, w_g)
            bvec = eta_k * tmp  # (n,)
            Kb = Kmat @ bvec  # (n,)

            # Solve once, reuse the factorization for speed
            # The posterior mean needs M^{-1} Kb; Hutchinson needs M^{-1} KZ.
            # Stacking both right-hand sides factorizes M a single time.
            if m_eff > 0:
                rhs = torch.cat([Kb.unsqueeze(-1), KZ], dim=1)  # (n, 1 + m_eff)
                sol = torch.linalg.solve(M, rhs)  # (n, 1 + m_eff)
                Ksol = Kmat @ sol  # (n, 1 + m_eff)
                mvec = Kb - Ksol[:, 0]  # (n,)
                # Hutchinson estimate of diag(S), S = K - K M^{-1} K
                SZ = KZ - Ksol[:, 1:]  # (n, m_eff)
                vdiag = (SZ * Z).mean(dim=1)  # (n,)
            else:
                y = torch.linalg.solve(M, Kb.unsqueeze(-1)).squeeze(-1)  # (n,)
                mvec = Kb - Kmat @ y  # (n,)
                # Exact but very expensive: S = K - K M^{-1} K
                X = torch.linalg.solve(M, Kmat)  # (n, n)
                S = Kmat - Kmat @ X
                vdiag = torch.diagonal(S)

            # Variances must be non-negative; clamp numerical undershoot to 0.
            vdiag = torch.clamp(vdiag, min=0.0)

            # Update fitted & residual only if mean changed
            mu_old = col
            delta = mvec - mu_old
            if torch.linalg.norm(delta) > 0:
                upd = torch.outer(delta * eta_k, theta_r)  # (n, p)
                fitted.add_(upd)
                R_base.sub_(upd)

            mu_Xi[:, r, k] = mvec
            var_Xi[:, r, k] = vdiag

    return mu_Xi, var_Xi


# ============================================
# CAVI loop
# ============================================


def cavi(Y, Kmat, init, max_iter=30, min_iter=10, tol=1e-3, verbose=True, m=100):
    """Run coordinate-ascent variational inference for the SCR model.

    Starting from ``init``, each iteration performs one full CAVI sweep,
    updating the blocks in the order

        eta  ->  Theta  ->  phi  ->  delta  ->  Xi  ->  sigma,

    each block to its evidence-lower-bound-optimal variational distribution
    given the others.

    Convergence is monitored by a relative-change metric
    ``agg = sqrt( (||d_theta||^2 + ||d_xi||^2 + ||d_sigma||^2)
                  / (||theta||^2 + ||xi||^2 + ||E[1/sigma^2]||^2) )``
    computed on the means of ``Theta``, ``Xi`` and on ``E[1/sigma^2]``. The
    loop stops once ``it >= min_iter`` and ``agg <= tol`` on three
    consecutive iterations, or when ``max_iter`` is reached.

    Parameters
    ----------
    Y : torch.Tensor, shape (n, p)
        Observations. Must live on the same device/dtype as the ``init``
        tensors.
    Kmat : torch.Tensor, shape (n, n)
        Spatial GP prior covariance over the ``n`` locations (e.g. from
        :func:`scr.utils.se_kernel`). Same device/dtype as ``init``.
    init : dict
        Initial variational state. Required keys:
        ``mu_Theta`` (p, L), ``Sigma_Theta_list`` (list of (L, L)),
        ``mu_eta`` (n, K), ``Sigma_eta_list`` (list of (K, K)),
        ``mu_Xi`` (n, L, K), ``var_Xi`` (n, L, K),
        ``a_phi`` / ``b_phi`` (p, L), ``a_delta`` / ``b_delta`` (L,),
        ``a_sigma`` / ``b_sigma`` (p,), ``a0`` / ``b0`` (scalar or (p,)),
        ``nu`` (float), and optionally ``prior_prec`` ((K, K), default
        identity). ``init["a_delta"]`` must be the **prior** shape of
        ``delta``; see :func:`update_delta`.
    max_iter : int, optional
        Maximum number of CAVI sweeps. Defaults to ``30``.
    min_iter : int, optional
        Minimum number of sweeps before the stopping rule may trigger.
        Defaults to ``10``.
    tol : float, optional
        Relative-change tolerance for the stopping rule. Defaults to ``1e-3``.
    verbose : bool, optional
        If ``True`` (default) print per-iteration diagnostics.
    m : int, optional
        Number of Hutchinson probe vectors passed to :func:`update_xi`.
        Defaults to ``100``.

    Returns
    -------
    dict
        ``{"params": params, "trace": trace}`` where

        * ``params`` holds the fitted variational parameters: ``mu_eta``,
          ``Sigma_eta_list``, ``mu_Theta``, ``Sigma_Theta_list``,
          ``a_phi`` / ``b_phi``, ``a_delta`` / ``b_delta``, ``mu_Xi`` /
          ``var_Xi``, ``a_sigma`` / ``b_sigma``;
        * ``trace`` holds per-iteration lists: ``iter`` and the block changes
          ``d_eta``, ``d_theta``, ``d_phi``, ``d_delta``, ``d_xi``,
          ``d_sigma``, plus ``max_delta``.

    Notes
    -----
    The convergence metric tracks only ``Theta``, ``Xi`` and
    ``sigma``, which are the blocks that determine the fitted covariance field. The
    remaining block changes are still recorded in ``trace`` for inspection.
    """
    with torch.no_grad():
        # unpack init
        mu_Theta = init["mu_Theta"]
        mu_eta = init["mu_eta"]
        Sigma_Theta_list = init["Sigma_Theta_list"]
        Sigma_eta_list = init["Sigma_eta_list"]
        mu_Xi = init["mu_Xi"]
        var_Xi = init["var_Xi"]
        a_sigma = init["a_sigma"]
        b_sigma = init["b_sigma"]
        a_phi = init["a_phi"]
        b_phi = init["b_phi"]
        a_delta = init["a_delta"]
        b_delta = init["b_delta"]
        a_delta_prior = init["a_delta"]
        nu = init["nu"]
        a0 = init["a0"]
        b0 = init["b0"]
        prior_prec = init.get("prior_prec", None)

        trace = {
            "iter": [],
            "L2_rel": [],
            "d_eta": [],
            "d_theta": [],
            "d_phi": [],
            "d_delta": [],
            "d_xi": [],
            "d_sigma": [],
            "max_delta": [],
        }

        # how many consecutive iters we've had agg_metric <= tol
        num_good = 0

        for it in range(1, max_iter + 1):
            mu_eta_old = mu_eta.clone()
            mu_Theta_old = mu_Theta.clone()
            mu_Xi_old = mu_Xi.clone()

            E_phi_old = a_phi / b_phi
            E_delta_old = a_delta / b_delta
            E_sig_old = a_sigma / b_sigma

            # eta
            mu_eta, Sigma_eta_list = update_eta(
                Y,
                mu_Theta,
                Sigma_Theta_list,
                mu_Xi,
                var_Xi,
                a_sigma,
                b_sigma,
                prior_prec=prior_prec,
            )

            # Theta
            mu_Theta, Sigma_Theta_list = update_Theta(
                Y,
                mu_eta,
                Sigma_eta_list,
                mu_Xi,
                var_Xi,
                a_phi,
                b_phi,
                a_delta,
                b_delta,
                a_sigma,
                b_sigma,
            )

            # phi
            a_phi, b_phi = update_phi(mu_Theta, Sigma_Theta_list, a_delta, b_delta, nu)

            # delta
            a_delta, b_delta = update_delta(
                mu_Theta, Sigma_Theta_list, a_phi, b_phi, a_delta_prior, b_delta
            )

            # xi
            mu_Xi, var_Xi = update_xi(
                Y,
                Kmat,
                mu_Theta,
                Sigma_Theta_list,
                mu_eta,
                Sigma_eta_list,
                mu_Xi,
                var_Xi,
                a_sigma,
                b_sigma,
                m=m,
                jitter=1e-8,
            )

            # sigma
            a_sigma, b_sigma = update_sigma(
                Y,
                mu_Theta,
                Sigma_Theta_list,
                mu_eta,
                Sigma_eta_list,
                mu_Xi,
                var_Xi,
                a0,
                b0,
            )

            d_eta = frob(mu_eta - mu_eta_old).item()
            d_theta = frob(mu_Theta - mu_Theta_old).item()
            d_xi = frob_array(mu_Xi, mu_Xi_old).item()

            E_phi = a_phi / b_phi
            E_delta = a_delta / b_delta
            E_sig = a_sigma / b_sigma

            d_phi = frob(E_phi - E_phi_old).item()
            d_delta = l2(E_delta - E_delta_old).item()
            d_sigma = l2(E_sig - E_sig_old).item()

            max_delta = max(d_eta, d_theta, d_xi, d_phi, d_delta, d_sigma)

            numerator = d_theta**2 + d_xi**2 + d_sigma**2
            theta_sqrt = frob(mu_Theta_old).item()
            xi_sqrt = torch.linalg.norm(mu_Xi_old).item()
            sig_sqrt = l2(E_sig_old).item()
            denominator = theta_sqrt**2 + xi_sqrt**2 + sig_sqrt**2
            agg_metric = math.sqrt(numerator / denominator) if denominator > 0 else 0.0

            if verbose:
                msg = (
                    f"iter {it}"
                    f" | L2_rel = {agg_metric:.3e}"
                    f" | max_d = {max_delta:.1e}"
                    f" | d_eta = {d_eta:.1e}"
                    f", d_theta = {d_theta:.1e}"
                    f", d_phi = {d_phi:.1e}"
                    f", d_delta = {d_delta:.1e}"
                    f", d_xi = {d_xi:.1e}"
                    f", d_sigma = {d_sigma:.1e}"
                )
                print(msg, flush=True)

            trace["iter"].append(it)
            trace["L2_rel"].append(agg_metric)
            trace["d_eta"].append(d_eta)
            trace["d_theta"].append(d_theta)
            trace["d_phi"].append(d_phi)
            trace["d_delta"].append(d_delta)
            trace["d_xi"].append(d_xi)
            trace["d_sigma"].append(d_sigma)
            trace["max_delta"].append(max_delta)

            # stopping rule: need 3 consecutive iters with agg_metric <= tol (and it >= min_iter)
            if it >= min_iter:
                if agg_metric <= tol:
                    num_good += 1
                else:
                    num_good = 0  # reset if condition is broken

                if num_good >= 3:
                    if verbose:
                        print(
                            f">> stopping at iter {it} because agg_metric <= tol "
                            f"for {num_good} consecutive iterations "
                            f"(agg_metric={agg_metric:.3e}, tol={tol}, min_iter={min_iter})"
                        )
                    break

        params = {
            "mu_eta": mu_eta,
            "Sigma_eta_list": Sigma_eta_list,
            "mu_Theta": mu_Theta,
            "Sigma_Theta_list": Sigma_Theta_list,
            "a_phi": a_phi,
            "b_phi": b_phi,
            "a_delta": a_delta,
            "b_delta": b_delta,
            "mu_Xi": mu_Xi,
            "var_Xi": var_Xi,
            "a_sigma": a_sigma,
            "b_sigma": b_sigma,
        }

        return {"params": params, "trace": trace}


# ============================================
# scr_init
# ============================================


def scr_init(
    Y,
    L,
    K,
    a_delta,
    b_delta,
    a0,
    b0,
    nu,
    spca_center=True,
    spca_alpha=1.0,
    spca_ridge_alpha=0.01,
    device=None,
    dtype=None,
):
    """
    The initialization proceeds in three stages:

    1. **Sparse PCA** of ``Y`` gives an initial loading
       matrix ``Theta`` (``p x L``) and scores; the scores are standardised
       column-wise, and the scaling is compensated for in ``Theta``. The first ``K``
       columns become the initial factor scores ``mu_eta``.
    2. **Per-spot least squares** reconstructs each ``Xi_i`` as the rank-one
       array ``z_i eta_i^T / ||eta_i||^2`` where ``z_i`` regresses ``Y[i]`` on
       ``Theta``; residual-based standard errors seed ``var_Xi``. The noise
       variances are initialised from the median, across spots, of the
       diagonal mismatch between the empirical covariance of ``Y`` and the
       low-rank reconstruction.
    3. One call each to :func:`update_Theta` and :func:`update_eta` to make
       the means and covariances mutually consistent.

    Parameters
    ----------
    Y : array_like, shape (n, p)
        Observations (a NumPy array, or anything ``numpy.asarray`` accepts).
    L : int
        Latent loading dimension.
    K : int
        Latent factor dimension. Must satisfy ``K <= L``.
    a_delta : array_like
        Prior shape of the multiplicative-gamma factors. Either length 2 --
        ``(a1, a2)``, expanded to ``[a1, a2, ..., a2]`` -- or length ``L``.
    b_delta : array_like
        Prior rate of the multiplicative-gamma factors. Either a scalar
        (broadcast to length ``L``) or length ``L``.
    a0, b0 : float
        Inverse-gamma prior shape and rate for the noise variances.
    nu : float
        Degrees-of-freedom hyperparameter of the local-shrinkage prior on
        ``phi``.
    spca_center : bool, optional
        If ``True`` (default) centre ``Y`` column-wise before sparse PCA.
    spca_alpha : float, optional
        L1 sparsity penalty for :class:`sklearn.decomposition.SparsePCA`.
        Defaults to ``1.0``.
    spca_ridge_alpha : float, optional
        Ridge penalty in the SparsePCA transform step. Defaults to ``0.01``.
    device : torch.device or None, optional
        Device for the returned tensors. ``None`` selects
        :data:`scr.utils.DEVICE` (CUDA if available, else CPU).
    dtype : torch.dtype or None, optional
        Dtype for the returned tensors. ``None`` selects
        :data:`scr.utils.DTYPE` (``float64``).

    Returns
    -------
    dict
        An ``init`` dictionary suitable for :func:`cavi`. In addition to the
        variational-parameter tensors it carries the hyperparameters ``a0``,
        ``b0``, ``nu`` and the ``(K, K)`` prior precision ``prior_prec``.
        ``init["a_delta"]`` is the **prior** shape of ``delta``.

    Raises
    ------
    ValueError
        If ``K > L``, or if ``a_delta`` / ``b_delta`` have an unsupported
        length.
    """
    if device is None:
        device = DEVICE
    if dtype is None:
        dtype = DTYPE

    Y = np.asarray(Y, dtype=float)
    N, p = Y.shape
    if K > L:
        raise ValueError("K must be <= L for this initialization (scores[:, :K]).")

    a_phi_np = np.full((p, L), nu / 2.0, dtype=float)
    b_phi_np = np.full((p, L), nu / 2.0, dtype=float)

    a_delta = np.asarray(a_delta, dtype=float)
    if a_delta.size == 2:
        a1, a2 = a_delta
        a_delta_np = np.empty(L, dtype=float)
        a_delta_np[0] = a1
        a_delta_np[1:] = a2
    elif a_delta.size == L:
        a_delta_np = a_delta
    else:
        raise ValueError("a_delta must be length-2 or length-L.")

    b_delta = np.asarray(b_delta, dtype=float)
    if b_delta.size == 1:
        b_delta_np = np.full(L, b_delta.item(), dtype=float)
    elif b_delta.size == L:
        b_delta_np = b_delta
    else:
        raise ValueError("b_delta must be scalar or length-L.")

    if spca_center:
        Y_center = Y - Y.mean(axis=0, keepdims=True)
    else:
        Y_center = Y

    spca = SparsePCA(
        n_components=L, alpha=spca_alpha, ridge_alpha=spca_ridge_alpha, n_jobs=-1
    )
    # spca = PCA(n_components=L, svd_solver='auto')
    scores = spca.fit_transform(Y_center)  # N x L
    Theta = spca.components_.T  # p x L

    for col in range(L):
        col_std = scores[:, col].std(ddof=1)
        if col_std < 1e-12:
            continue
        scores[:, col] /= col_std
        Theta[:, col] *= col_std

    mu_Theta_np = Theta  # p x L
    mu_eta_np = scores[:, :K]  # N x K

    Y_GN = Y.T  # p x N

    AtA = mu_Theta_np.T @ mu_Theta_np  # L x L
    TtY = mu_Theta_np.T @ Y_GN  # L x N

    try:
        L_AtA = np.linalg.cholesky(AtA)
        AtA_inv = np.linalg.solve(L_AtA.T, np.linalg.solve(L_AtA, np.eye(L)))
    except np.linalg.LinAlgError:
        AtA_inv = np.linalg.inv(AtA)

    mu_Xi_np = np.zeros((N, L, K), dtype=float)
    var_Xi_np = np.zeros((N, L, K), dtype=float)

    rank_Theta = np.linalg.matrix_rank(mu_Theta_np)

    for i in range(N):
        etai = mu_eta_np[i, :]  # K
        e2 = np.dot(etai, etai)
        if e2 < 1e-12:
            continue

        zi = AtA_inv @ TtY[:, i]  # L

        resid = Y_GN[:, i] - mu_Theta_np @ zi  # p
        # guard against p == rank(Theta) (e.g. p == L) -> zero denominator
        sig2_i = np.dot(resid, resid) / max(p - rank_Theta, 1)

        var_z_diag = sig2_i * np.diag(AtA_inv)  # L

        Xi_i = np.outer(zi, etai) / e2
        mu_Xi_np[i, :, :] = Xi_i

        se_r = np.sqrt(var_z_diag)  # L
        se_mat = np.outer(se_r, np.abs(etai) / e2)
        var_Xi_np[i, :, :] = se_mat**2

    YTY = np.cov(Y, rowvar=False)  # p x p

    diff_list = []
    for i in range(N):
        Lambda_i = mu_Theta_np @ mu_Xi_np[i, :, :]  # p x K
        LambdaLambdaT = Lambda_i @ Lambda_i.T  # p x p
        diff_diag = np.diag(YTY - LambdaLambdaT)  # p
        diff_list.append(diff_diag)

    qq = np.vstack(diff_list)  # N x p
    sigma_est = np.median(qq, axis=0)  # p

    sigma_est = np.maximum(sigma_est, 1e-8)

    a_sigma_scalar = a0 + N / 2.0
    a_sigma_np = np.full(p, a_sigma_scalar, dtype=float)

    b_sigma_np = a_sigma_np * sigma_est

    Y_torch = torch.as_tensor(Y, device=device, dtype=dtype)
    mu_Theta_t = torch.as_tensor(mu_Theta_np, device=device, dtype=dtype)
    mu_eta_t = torch.as_tensor(mu_eta_np, device=device, dtype=dtype)
    mu_Xi_t = torch.as_tensor(mu_Xi_np, device=device, dtype=dtype)
    var_Xi_t = torch.as_tensor(var_Xi_np, device=device, dtype=dtype)
    a_phi_t = torch.as_tensor(a_phi_np, device=device, dtype=dtype)
    b_phi_t = torch.as_tensor(b_phi_np, device=device, dtype=dtype)
    a_delta_t = torch.as_tensor(a_delta_np, device=device, dtype=dtype)
    b_delta_t = torch.as_tensor(b_delta_np, device=device, dtype=dtype)
    a_sigma_t = torch.as_tensor(a_sigma_np, device=device, dtype=dtype)
    b_sigma_t = torch.as_tensor(b_sigma_np, device=device, dtype=dtype)
    a0_t = torch.tensor([float(a0)], device=device, dtype=dtype)
    b0_t = torch.tensor([float(b0)], device=device, dtype=dtype)
    prior_prec_t = torch.eye(K, device=device, dtype=dtype)

    Sigma_eta_list = [torch.eye(K, device=device, dtype=dtype) for _ in range(N)]

    mu_Theta_t, Sigma_Theta_list = update_Theta(
        Y_torch,
        mu_eta_t,
        Sigma_eta_list,
        mu_Xi_t,
        var_Xi_t,
        a_phi_t,
        b_phi_t,
        a_delta_t,
        b_delta_t,
        a_sigma_t,
        b_sigma_t,
    )

    mu_eta_t, Sigma_eta_list = update_eta(
        Y_torch,
        mu_Theta_t,
        Sigma_Theta_list,
        mu_Xi_t,
        var_Xi_t,
        a_sigma_t,
        b_sigma_t,
        prior_prec=prior_prec_t,
    )

    init = {
        "mu_eta": mu_eta_t,
        "Sigma_eta_list": Sigma_eta_list,
        "mu_Theta": mu_Theta_t,
        "Sigma_Theta_list": Sigma_Theta_list,
        "a_phi": a_phi_t,
        "b_phi": b_phi_t,
        "a_delta": a_delta_t,
        "b_delta": b_delta_t,
        "mu_Xi": mu_Xi_t,
        "var_Xi": var_Xi_t,
        "a_sigma": a_sigma_t,
        "b_sigma": b_sigma_t,
        "a0": a0_t,
        "b0": b0_t,
        "prior_prec": prior_prec_t,
        "nu": float(nu),
    }

    return init
