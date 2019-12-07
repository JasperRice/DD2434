import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.optimize as opt

from math import sin, cos, pi
from numpy import trace as tr
from numpy.linalg import det
from numpy.linalg import inv
from numpy.linalg import pinv

np.random.seed(2222)

def f_nonlin(X):
    return np.transpose(np.array([np.sin(X) - X * np.cos(X), np.cos(X) + X * np.sin(X)]))

def f_lin(X, A):
    return np.dot(X, np.transpose(A))

def _A_():
    return np.random.normal(size=20).reshape(10, 2)

def _X_(N):
    return np.linspace(0.0, 4*pi, num=N)

def _Y_(X, A):
    return f_lin(f_nonlin(X), A)

def f(W, *args):
    N, Y, sigma = args
    _W_ = W.reshape(10, 2)

    T0 = np.dot(_W_, np.transpose(_W_)) + sigma**2 * np.ones_like(np.dot(_W_, np.transpose(_W_)))
    T1 = np.dot(np.dot(Y, pinv(T0)), np.transpose(Y))
    return N * det(T0) / 2 + tr(T1) / 2

def gradf(W, *args):
    N, Y, sigma = args
    _W_ = W.reshape(10, 2)
    
    T0 = np.dot(_W_, np.transpose(_W_)) \
        + sigma**2 * np.ones_like(np.dot(_W_, np.transpose(_W_)))
    T1 = np.dot(np.transpose(Y), Y)
    return (N * np.dot(pinv(T0), _W_) - np.dot(np.dot(np.dot(pinv(T0), T1), pinv(T0)), _W_)).reshape(-1)

def quesion_16():
    # NList = [50, 100, 200, 500]
    N = 200
    X = _X_(N)
    W0 = _A_()
    Y = _Y_(X, W0)
    sigma = 1

    args = (N, Y, sigma)
    W_Hat = opt.fmin_cg(f, W0.reshape(-1), fprime=gradf, args=args).reshape(10, 2)
    X_Hat = np.dot(np.dot(Y, W_Hat), inv(np.dot(np.transpose(W_Hat), W_Hat)))

    # X_Hat = 5.7 * X_Hat

    fig, ax = plt.subplots()
    ax.plot(f_nonlin(X)[:, 0], f_nonlin(X)[:, 1])
    ax.plot(X_Hat[:, 0], X_Hat[:, 1])
    ax.grid()
    ax.set_aspect('equal', 'box')
    plt.xlabel("$x_{0}^{'}$")
    plt.ylabel("$x_{1}^{'}$")
    plt.show()


if __name__ == "__main__":
    quesion_16()