from math import exp, pi, sqrt
from scipy.special import gamma

import math
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1111)
MAX_ITER = 100
THRES = 0.001

def generateData(N, mu=0.0, sigma=1.0):
	D = np.random.normal(mu, sigma, N)
	return(D)

def _plot(mu, tau, p, q, color):
	m, t = np.meshgrid(mu, tau)
	plt.figure()
	plt.contour(m, t, p)
	plt.contour(m, t, q, colors=color)
	plt.xlabel('$\\mu$')
	plt.ylabel('$\\tau$')
	plt.axis("equal")

# True posterior distribution
def _p(muT, lambdaT, aT, bT, mu, tau):
	p = (bT**aT)*sqrt(lambdaT) / (gamma(aT)*sqrt(2*pi)) * tau**(aT-0.5) \
		* np.exp(-bT*tau) * np.exp(-0.5*lambdaT*np.dot(tau,((mu-muT)**2).T))
	return p

# Approximated posterior distribution
def _q(muN, lambdaN, aN, bN, mu, tau):
    q_mu = sqrt(lambdaN/(2*pi)) * np.exp(-0.5 * np.dot(lambdaN, np.transpose((mu-muN)**2)))
    q_tau = (1.0/gamma(aN)) * bN**aN * tau**(aN-1) * np.exp(-bN*tau)
    q = q_tau * q_mu
    return q

# Update parameter
def _update(D, N, mu0, lambda0, a0, b0, muN, lambdaN, aN, bN):
    E_mu = muN
    E_mu2 = 1.0 / lambdaN + muN**2
    E_tau = aN / bN
    lambdaN = (lambda0 + N) * E_tau
    bN = b0 - (sum(D) + lambda0*mu0)*E_mu \
        + 0.5*(sum(D**2) + lambda0*mu0**2 + (lambda0+N)*E_mu2)
    return lambdaN, bN

def simpleVI():
	N = 10
	D = generateData(N)
	x_bar = D.mean()
	mu 	= np.linspace(-2, 2, 100)
	tau = np.linspace( 0, 4, 100)
	a0 = 0
	b0 = 0
	mu0 = 0
	lambda0 = 0
	
	muT = (lambda0 * mu0 + N * x_bar) / (lambda0 + N)
	lambdaT = lambda0 + N
	aT = a0 + N/2
	bT = b0 + 0.5*sum((D-x_bar)**2) + (lambda0*N*(x_bar-mu0)**2)/(2*(lambda0+N))
	
	muN = (lambda0 * mu0 + N * x_bar) / (lambda0 + N)
	lambdaN = 0.1
	aN = a0 + (N + 1) / 2
	bN = 0.1
	lambdaOld = lambdaN
	bOld = bN

	p = _p(muT, lambdaT, aT, bT, mu[:,None], tau[:,None])

	for iter in range(MAX_ITER):
		lambdaN, bN = _update(D, N, mu0, lambda0, a0, b0, muN, lambdaN, aN, bN)
		q = _q(muN, lambdaN, aN, bN, mu[:,None], tau[:,None])

		if (abs(lambdaN - lambdaOld) < THRES) and (abs(bN - bOld) < THRES):
			_plot(mu, tau, p, q, 'r')
			plt.savefig("Q2_4_1_N"+str(N)+"_i"+str(iter))
			break
		else:
			_plot(mu, tau, p, q, 'b')
			plt.savefig("Q2_4_1_N"+str(N)+"_i"+str(iter))
		lambdaOld = lambdaN
		bOld = bN

	print(muN, lambdaN, aN, bN)


if __name__ == "__main__":
	simpleVI()