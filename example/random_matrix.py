import torch
from lib import svp

M = 10
N = 20
rank = 2

gt = torch.rand((1, M, N))
U, S, Vh = torch.linalg.svd(gt,full_matrices=False)
S[:, rank:] = 0
gt = U @ torch.diag_embed(S) @ Vh

indices = torch.randperm(M * N)[: int(0.6 * M * N)]  # 20% elements are observed
mask = torch.zeros(M * N, dtype=torch.int32)
mask[indices] = 1
mask = mask.reshape([M, N])
observed_matrix = mask[None] * gt

problem = dict(
    channel = 1,
    size=[M, N],
    gt=gt,
    rank=rank,
    observed_matrix=observed_matrix,
    mask=mask,
    method=svp,
    step=1,
    tol=1e-4,
    device="cuda:0"
)

del(torch)