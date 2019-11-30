import matplotlib.pyplot as plt
import tikzplotlib as tikz
import numpy as np
import pylab as pb
from math import pi
from scipy.spatial.distance import cdist
from scipy.stats import multivariate_normal as mvn

W = np.array([0.5, -1.5])
# W = np.transpose([W])

X = np.linspace(-1.0, 1.0, num=201)
X = np.vstack([X, np.ones_like(X)])
# X = np.transpose(X)

f = np.dot(W, X)
# f = np.dot(X, W)

def plot_2d_gaussian(mu, Sigma, size=3.0):
    rv = mvn(mu, Sigma)

    x, y = np.mgrid[mu[0]-size : mu[0]+size : 0.01,
                    mu[1]-size : mu[1]+size : 0.01]
    X = np.dstack([x, y])

    _, ax = plt.subplots()
    CS = ax.contour(x, y, rv.pdf(X))
    # CS = ax.contourf(x, y, rv.pdf(X))
    ax.clabel(CS, inline=0.5, fontsize=10)
    ax.set_aspect('equal', 'box')
    tikz.save("2d_guassian.tex")

def question_9():
    # 1
    mu_w = np.zeros(2)
    Sigma_w = np.eye(2)
    plot_2d_gaussian(mu_w, Sigma_w)

    # 2


if __name__ == "__main__":
    question_9()