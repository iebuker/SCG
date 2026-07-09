# SCG: Spatially Co-expressed Gene Identification

## Model

Observations `Y` (`n` spots × `p` genes) are modelled as

```
Y[i, :] = Theta @ Xi_i @ eta_i + epsilon_i
```

where `Theta` (`p` × `L`) are global gene loadings, `Xi_i` (`L` × `K`) are
spatially varying coefficients at spot `i` drawn from independent Gaussian
process priors, and `eta_i` (`K`,) are latent factor scores. Inference is
performed via coordinate-ascent variational inference (CAVI) under a
mean-field approximation.

## Installation

```bash
pip install git+https://github.com/iebuker/scr-python
```

Requires: 

- numpy>=1.23
- scikit-learn>=1.1
- torch>=2.0
- pandas>=1.4

## Quick start

```python
import torch
import scr

scr.set_seed(0)

# 1. data + spatial coordinates
Y, S, info = scr.simulate_data_cov(N=500, p=30, seed=0)

# 2. spatial GP prior covariance over the locations
Kmat = scr.rq_kernel(S, rho=0.3, alpha=1.0)

# 3. sparse-PCA warm start
init = scr.scr_init(Y, L=7, K=5, a_delta=(2.1, 3.1),
                    b_delta=1.0, a0=2.0, b0=1.0, nu=3.0)

# 4. move data + kernel onto the device/dtype, then run CAVI
Y_t = torch.as_tensor(Y, device=scr.DEVICE, dtype=scr.DTYPE)
K_t = torch.as_tensor(Kmat, device=scr.DEVICE, dtype=scr.DTYPE)
out = scr.cavi(Y_t, K_t, init, max_iter=40)
```

See https://iebuker.github.io/SCG/ for a full worked example in Python and R, along with installation instructions.

## Authors

Ihsan E. Buker (iebuker@uab.edu) and Satwik Acharyya (acharyya@uab.edu)

## License

MIT
