import torch

@torch.no_grad()
def svp(observed_matrix,mask,step,k,maxIter,tol, status={}):
    '''
    observed_matrix: (C, M, N) with only a few observations
    mask: (M, N)
    '''
    r0 = (observed_matrix).abs().max()
    X = torch.zeros_like(observed_matrix)
    for i in range(maxIter):
        g = mask[None] * (X - observed_matrix)
        res = g.abs().max() / r0
        print(f"Iter: {i}, residual: {res}")
        if res  < tol:
            break
        Y = X - step * g
        U, S, Vh = torch.linalg.svd(Y, full_matrices=False)
        X = U[:, :, :k] @ torch.diag_embed(S[:, :k]) @ Vh[:, :k, :]
    status["iter"] = i
    return X


    