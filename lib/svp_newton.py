import torch
import numpy as np
from scipy.optimize import least_squares

def fun_rosenbrock(x, u, vh, ob, mk):
    [M, N, k] = [u.shape[0], vh.shape[1], u.shape[1]]
    x = x.reshape(k, k)
    x = mk * (u @ x @ vh - ob)
    return x.reshape(M * N, )


@torch.no_grad()
def svp_newton(observed_matrix,mask,step,k,maxIter,tol):
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

        ####
        x0 = np.zeros(k * k)
        u = np.array(U[0, :, 0:k])
        v = np.array(Vh[0, 0:k, :])
        ob = np.array(observed_matrix[0, :, :])
        mk = np.array(mask)
        res = least_squares(fun_rosenbrock, x0, args=(u, v, ob, mk))

        xx = res.x.reshape(k, k)
        xx = u @ xx @ v
        xx = torch.from_numpy(xx)
        X = torch.reshape(xx, X.shape)

        # S[:, k:] = 0
        # X = U @ torch.diag_embed(S) @ Vh
    return X
