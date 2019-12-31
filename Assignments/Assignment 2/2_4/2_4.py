_author_ = "Sifan Jiang"

from math import exp, pi, sqrt
from scipy.special import gamma

import math
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1000)
MAXITER = 1000

def _plot(mu, tau, p):
    pass

def _p(muT, lambdaT, aT, bT, mu, tau):
    pass

def _q(muN, lambdaN, aN, bN, mu, tau):
    q_mu = sqrt(lambdaN/(2*pi)) * np.exp(-0.5 * np.dot(lambdaN, np.transpose((mu-muN)**2)))
    q_tau = (1.0/gamma(aN)) * bN**aN * tau**(aN-1) * np.exp(-bN*tau)
    q = q_tau * q_mu
    return q

def _update(D, N, mu0, lambda0, a0, b0, muN, lambdaN, aN, bN):
    E_mu = muN
    E_mu2 = 1.0 / lambdaN + muN**2
    E_tau = aN / bN
    lambdaN = (lambda0 + N) * E_tau
    bN = b0 - (sum(D) + lambda0*mu0)*E_mu \
        + 0.5*(sum(D**2) + lambda0*mu0**2 + (lambda0+N)*E_mu2)
    return lambdaN, bN

def SimpleVI():
    # Generate data set.
    mu_D = 0.0
    sigma_D = 1.0
    N = 100
    D = np.random.normal(loc=mu_D, scale=sigma_D, size=N)

    # Initial values.
    x_bar = D.mean()
    mu0 = 0
    lambda0 = 0
    a0 = 0
    b0 = 0

    muN = (lambda0*mu0 + N * x_bar) / (lambda0 + N)
    lambdaN = 10
    aN = a0 + N / 2
    bN = 5

    mu = np.linspace(mu_D-2, mu_D+2, num=1000)
    tau = np.linspace(sigma_D-2, sigma_D+2, num=1000)

    # p = _p(muT, lambdaT, aT, bT, mu, tau)

    for i in range(MAXITER):
        lambdaN, bN = _update(D, N, mu0, lambda0, a0, b0, muN, lambdaN, aN, bN)
        q = _q(muN, lambdaN, aN, bN, mu, tau)

    print(muN, lambdaN, aN, bN)

if __name__ == "__main__":
    SimpleVI()