import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import scr_model as sm

seed = 123
cli_arg = sys.argv[1]

base_path = Path("../ad_data") / cli_arg / "data_cleaned"
counts_name = "X_norm.csv"
fit_name = "fit_0.75l.pt"
loc_name = "S.csv"

print("cli_arg:", cli_arg)
print("counts_name:", counts_name)
print("fit_name:", fit_name)
print("base_path:", base_path)


Y_np = pd.read_csv(base_path / counts_name).iloc[:, 1:].to_numpy().T
# This location matrix read in with a third column 
S = pd.read_csv(base_path / loc_name).iloc[:, 1:].to_numpy()
G = Y_np.shape[1]
kl_val = np.round(2 * np.log(G)).astype(int)
init = sm.scr_init(
    Y=Y_np,
    L=kl_val,
    K=kl_val,
    a_delta=(2.1, 3.1),
    b_delta=1.0,
    a0=5.0,
    b0=0.5,
    nu=5.0,
    spca_center=True,
    spca_alpha=0.01,
    spca_ridge_alpha=0.01,
    device=sm.DEVICE,
    dtype=sm.DTYPE,
)

kernel_hypers = sm.pick_kernel_params(init=init, S=S, score_quantile=0.9)
print(kernel_hypers)
K_np = sm.rq_kernel(
    S,
    v2=1.0,
    l= 0.75 * kernel_hypers["l"],
    alpha=kernel_hypers["alpha"],
)


# ----------------------------
# Convert to torch tensors
# ----------------------------
Y = torch.as_tensor(Y_np, device=sm.DEVICE, dtype=sm.DTYPE)
K = torch.as_tensor(K_np, device=sm.DEVICE, dtype=sm.DTYPE)

# ----------------------------
# Run CAVI
# ----------------------------
print("[INFO] Starting CAVI...")
cavi_start = time.time()
fit = sm.cavi(
    Y,
    K,
    init,
    max_iter=30,
    min_iter=10,
    tol=1e-3,
    verbose=True,
    hutch_m=128,
)
cavi_elapsed = time.time() - cavi_start
print(f"[INFO] CAVI finished in {cavi_elapsed:.2f} seconds.")

# Save fit object
torch.save(fit, base_path / fit_name)


