import time
import torch
from lib import svp, svp_newton
# iterate n

torch.random.manual_seed(123)
device = torch.device("cuda:0")

running_time_svp = []
running_time_svp_newton = []
iteration_svp = []
iteration_svp_newton = []
N = 1000
for k in torch.linspace(1, 100, 10):
    # prepare data
    tol = 1e-3
    k = int(k)
    p = 0.3
    indices = torch.randperm(N * N)[: int(p * N * N)]  # 20% elements are observed
    mask = torch.zeros(N * N, dtype=torch.int32)
    mask[indices] = 1
    mask = mask.reshape([N, N])
    gt = torch.rand((1, N, N))
    U, S, Vh = torch.linalg.svd(gt,full_matrices=False)
    S[:, k:] = 0
    gt = U @ torch.diag_embed(S) @ Vh
    observed_matrix = mask[None] * gt

    observed_matrix = observed_matrix.to(device)
    mask = mask.to(device)
    step = 0.1
    tol = 1e-3

    # run svp
    start = time.time()
    status = {}
    X_recon = svp(observed_matrix, mask, step, k, 10000, tol, status)
    error = (gt - X_recon.cpu()).abs().max() / gt.abs().max()
    print("svp:", f"k: {k}", status["iter"], error.item())
    cost = time.time() - start
    running_time_svp.append(cost)
    iteration_svp.append(status["iter"])
    

    # run svp-Newton
    start = time.time()
    status = {}
    X_recon = svp_newton(observed_matrix, mask, step, k, 1000, tol, status)
    error = (gt - X_recon.cpu()).abs().max() / gt.abs().max()
    print("svp-newton:", f"k: {k}", status["iter"], error.item())
    cost = time.time() - start
    running_time_svp_newton.append(cost)
    iteration_svp_newton.append(status["iter"])

import os
import numpy as np
os.makedirs("plot_data", exist_ok=True)
with open('plot_data/k_svp_time.npy', 'wb') as f:
    np.save(f, np.array(running_time_svp))
with open('plot_data/k_svp_newton_time.npy', 'wb') as f:
    np.save(f, np.array(running_time_svp_newton))
with open('plot_data/k_svp_iteration.npy', 'wb') as f:
    np.save(f, np.array(iteration_svp))
with open('plot_data/k_svp_newton_iteration.npy', 'wb') as f:
    np.save(f, np.array(iteration_svp_newton))
    
    