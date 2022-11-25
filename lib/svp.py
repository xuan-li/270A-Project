import torch

@torch.no_grad()
def svp(observed_matrix,mask,step,k,maxIter,tol):
    '''
    observed_matrix: (C, M, N) with only a few observations
    mask: (M, N)
    '''
    X = torch.zeros_like(observed_matrix)
    for i in range(maxIter):
        g = mask[None] * (X - observed_matrix)
        res = torch.abs(g).max()
        print(f"Iter: {i}, residual: {res}")
        if res  < tol:
            break
        Y = X - step * g
        U, S, Vh = torch.linalg.svd(Y, full_matrices=False)
        S[:, k:] = 0
        X = U @ torch.diag_embed(S) @ Vh
    return X


    