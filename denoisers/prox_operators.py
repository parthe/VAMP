# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 18:38:57 2021

@author: Parthe
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv, norm
eps = 1e-16


def prox_l1(r, gamma, Lambda=1.):
    # returns  argmin_x gamma/2 * ||x-r||^2 + Lambda * ||x||_1
    
    xhat = np.where(r>0,np.maximum(0,r-Lambda/gamma), np.minimum(0, r+Lambda/gamma))
    alpha = (np.count_nonzero(xhat)/len(xhat))
    eta = gamma/(eps+alpha)
    
    return xhat, eta, alpha

def prox_lp(r, gamma, Lambda=1, p=2, max_iter=3):
    
    """
    Returns argmin_x gamma/2 * ||x-r||^2 + Lambda / p * ||x||_p^p for p>1
    using Newtons method for ``max_iter`` iterations starting from 0
    For p=1 use ``prox_l1``
    For p=2 use ``prox_ridge`` for a faster implementation
    """
    xhat = np.zeros_like(r)
    for i in range(max_iter):
        grad = (gamma*(xhat-r)+Lambda*np.power(abs(xhat),(p-1)) *np.sign(xhat))
        if p>=2:
            hess_inv = 1/(gamma+Lambda*(p-1)*np.power(abs(xhat),(p-2)))
        else:
            hess_inv = 1/(gamma+Lambda*(p-1)/(eps+np.power(abs(xhat),(2-p))))
        xhat = xhat - grad*hess_inv # Newton iteration
        hess_inv_mean = hess_inv.mean()
    alpha = gamma * hess_inv_mean
    eta = gamma/alpha
    return xhat, eta, alpha

def prox_elastic_net(r, gamma, Lambda=1., l1_ratio=.5):
    """
    Returns argmin_x gamma/2* ||x-r||^2 + l1_ratio*Lambda* ||x||_1 + (1-l1_ratio)*Lambda/2* ||x||^2
    """
    xhat, _ , alpha = prox_l1(r, gamma, Lambda*l1_ratio)
    xhat = xhat * 1/(1 + Lambda/gamma * (1-l1_ratio))
    alpha = alpha * 1/(1 + Lambda/gamma * (1-l1_ratio))
    eta = gamma/(eps+alpha)
    return xhat, eta, alpha

def prox_ridge(r, gamma, Lambda=1):
    # returns  argmin_x gamma/2 * ||x-r||^2 + 1/2 * Lambda * ||x||^2
    alpha = (gamma/(Lambda+gamma))
    eta = gamma/alpha
    xhat = alpha * r 
    return xhat, eta, alpha


    
def pad(a,size):
    out=np.zeros(size)
    out[:len(a)] = a
    return out

def LMMSE_denoiser(r_2, gamma_2, noise_precision=0,A=0,y=0):
    # global variables A, y, noise_precision are assumed to be defined
    
    if type(A) is tuple:
        # SVD provided
        U, S, VT = A
        n,p=U.shape[0],VT.shape[1]
        S_v = pad(S,VT.shape[0]) # compatible with V
        S_u = pad(S,U.shape[1]) # compatible with U
        alpha = gamma_2 * sum(1/(noise_precision * S_v**2+ gamma_2 ))/p
        eta_2 = gamma_2/alpha
        matrix = (VT.T/(noise_precision*S_v**2+gamma_2))
        xhat_2 = matrix @( noise_precision *(pad(((y.T @U) * S_u).T,(p,1)))  +  gamma_2*VT@r_2 )
    else:
        matrix = inv(noise_precision * A.T @ A + gamma_2 * np.eye(A.shape[1]))
        
        xhat_2 = matrix @ (noise_precision * A.T@ y + gamma_2 * r_2)
        alpha = gamma_2 * np.trace(matrix)/matrix.shape[1]
        eta_2 = gamma_2/alpha
    
    return xhat_2, eta_2, alpha

