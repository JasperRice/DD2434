import matplotlib.pyplot as plt
import tikzplotlib as tikz
import numpy as np
import pylab as pb
import random
from math import pi
from scipy.spatial.distance import cdist
from scipy.stats import multivariate_normal as mvn

W = np.array([[0.5], [-1.5]])
X = np.linspace(-1.0, 1.0, num=201)

epsilon_sigma = 0.2


def plot_2d_gaussian(mu, Sigma, size=3.0, interval=0.01, filename="2d_guassian"):
    rv = mvn(mu, Sigma)
    x, y = np.mgrid[0-size : 0+size : interval, 0-size : 0+size : interval]
    xy = np.dstack([x, y])

    _, ax = plt.subplots()
    CS = ax.contour(x, y, rv.pdf(xy))
    # CS = ax.contourf(x, y, rv.pdf(X))
    ax.clabel(CS, inline=0.5, fontsize=8)
    ax.set_aspect('equal', 'box')
    tikz.save(filename+".tex")

def linear_function(W, x, epsilon):
    return np.dot(x, W) + epsilon

def generate_data_point():
    epsilon = np.array([[np.random.normal(0.0, epsilon_sigma)]])
    x = np.array([[random.choice(X), 1.0]])
    t = linear_function(W, x, epsilon)
    return x, t


def question_9(i):
    mu_prior = np.zeros(2)
    Sigma_prior = np.eye(2)
    Sigma_prior_inverse = np.linalg.inv(Sigma_prior)
    if i is 1:
        plot_2d_gaussian(mu_prior, Sigma_prior, filename="Q9-1")
    elif i is 2 or 3:
        x, t = generate_data_point()
        Sigma_posterior_inverse = np.dot(np.transpose(x), x) / (epsilon_sigma**2) + Sigma_prior_inverse
        Sigma_posterior = np.linalg.inv(Sigma_posterior_inverse)
        mu_posterior = np.dot(Sigma_posterior, np.transpose(x)) * t / (epsilon_sigma**2)
        mu_posterior = np.transpose(mu_posterior)[0,:]
        if i is 2:
            plot_2d_gaussian(mu_posterior, Sigma_posterior, filename="Q9-2")

        for n in range(6):
            
            pass

if __name__ == "__main__":
    question_9(2)