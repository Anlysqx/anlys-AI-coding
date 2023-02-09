import os
import numpy as np 

def alpha_forward(y, labels):
    # labels 是插入blank 后的
    T, V = y.shape # T: time step, V: probs
    L = len(labels)
    alpha = np.zeros([T, L])
    alpha[0, 0] = y[0, labels[0]]
    alpha[0, 1] = y[1, labels[1]]
    for t in range(1, T):
        for i in range(L):
            s = labels[i]
            a = alpha[t-1, i]
            if i - 1 >= 0:
                a += alpha[t-1, i-1]
            if i - 2 >= 0 and s != 0 and s != labels[i-2]:
                a += alpha[t-2, i-2]
            alpha[t, i] = a * y[t, s]
    return alpha

def beta_backward(y, labels):
    T, V = y.shape
    L = len(labels)
    beta = np.zeros([T, L])
    beta[-1, -1] = y[-1, labels[-1]]
    beta[-1, -2] = y[-1, labels[-2]]
    for t in range(T-2, -1, -1):
        for i in range(L):
            s = labels[i]
            b = beta[t+1, i]
            if i + 1 < L:
                b += beta[t+1, i+1]
            if i + 2 < l and s != 0 and s != labels[i + 2]:
                b += beta[t+2, i+2]
            beta[t, i] = b * y[t, s]
    return beta

def grad_backward(y, labels):
    T, V = y.shape
    L = len(labels)
    alpha = alpha_forward(y, labels)
    beta = beta_backward(y, labels)
    grad = np.zeros([T, V])
    probs = alpha[-1, -1] + alpha[-1, -2]
    for t in range(T):
        for s in range(V):
            labs = [i for i, c in enumerate(labels) if c == s]
            for i in labels:
                grad[t, s] += alpha[t, i] * beta[t, i]
            grad[t, s] /= y[t, s] ** 2
    grad /= probs
    return grad