import torch
import numpy as np
from scipy.optimize import least_squares

def loss_func(x, u, vh, ob, mk):
    res = mk[None] * (u @ x @ vh - ob)
    loss = (res ** 2).sum()
    return loss

def lbfgs_least_square(x, u, vh, ob, mk):
    x = x.clone().requires_grad_()
    optimizer = torch.optim.LBFGS([x], max_iter=10000, lr=1, tolerance_change=1e-20, tolerance_grad=1e-4)
    losses = []
    def closure():
        optimizer.zero_grad()
        loss = loss_func(x, u, vh, ob, mk)
        loss.backward()
        losses.append(loss.clone())
        return loss
    optimizer.step(closure)
    return x.data

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
        S0 = torch.diag_embed(S[:, :k])
        u = U[:, :, 0:k]
        vh = Vh[:, 0:k, :]
        ob = observed_matrix[:, :, :] 
        xx = lbfgs_least_square(S0, u, vh, ob, mask)
        xx = u @ xx @ vh
        X = xx.reshape(X.shape)

    return X
