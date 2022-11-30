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

def energy(X, observed_matrix, mask):
    return ((mask[None] * (observed_matrix - X)) ** 2).sum()

@torch.no_grad()
def svp_newton(observed_matrix,mask,step,k,maxIter,tol, status={}):
    '''
    observed_matrix: (C, M, N) with only a few observations
    mask: (M, N)
    '''
    X = torch.zeros_like(observed_matrix)
    r0 = (observed_matrix).abs().max()
    for i in range(maxIter):
        g = mask[None] * (X - observed_matrix)
        res = g.abs().max() / r0
        print(f"Iter: {i}, residual: {res}")
        if res  < tol:
            break
        E0 = energy(X, observed_matrix, mask)
        E1 = 1e30
        alpha = 1
        
        while E1 > E0:
            Y = X - alpha * g
            U, S, Vh = torch.linalg.svd(Y, full_matrices=False)
            S0 = torch.diag_embed(S[:, :k])
            u = U[:, :, 0:k]
            vh = Vh[:, 0:k, :]
            ob = observed_matrix[:, :, :] 
            xx = lbfgs_least_square(S0, u, vh, ob, mask)
            xx = u @ xx @ vh
            E1 = energy(xx, observed_matrix, mask)
            alpha /= 2
        
        X = xx.reshape(X.shape)
    status["iter"] = i
    return X
