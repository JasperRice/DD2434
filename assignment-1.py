import matplotlib.pyplot as plt
import tikzplotlib as tikz
import numpy as np
import pylab as pb
import random
from math import pi
from scipy.spatial.distance import cdist
from scipy.stats import multivariate_normal as mvn

W = np.array([[0.5], [-1.5]])
X_Set = np.linspace(-1.0, 1.0, num=201)
epsilon_sigma = 0.2

def plot_2d_gaussian(mu, Sigma, size=3.0, interval=0.01, filename="2d_guassian", save=True):
    rv = mvn(mu, Sigma)
    x, y = np.mgrid[0-size : 0+size : interval, 0-size : 0+size : interval]
    xy = np.dstack([x, y])

    _, ax = plt.subplots()
    CS = ax.contour(x, y, rv.pdf(xy))
    # CS = ax.contourf(x, y, rv.pdf(X))
    ax.clabel(CS, inline=0.5, fontsize=8)
    ax.set_aspect('equal', 'box')
    if save:
        tikz.save(filename+".tex")
    else:
        plt.show()

def plot_function(mu, Sigma):
    W = np.random.multivariate_normal(mu, Sigma)
    u"""
    Working here.
    """
    _, ax = plt.subplots()
    ax.axis('equal')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)

def linear_function(W_Model, x, epsilon):
    return np.dot(x, W_Model) + epsilon

def generate_data_point():
    epsilon = np.array([[np.random.normal(0.0, epsilon_sigma)]])
    x = np.array([[random.choice(X_Set), 1.0]])
    t = linear_function(W, x, epsilon)
    return x, t

def get_posterior(X, T, Sigma_prior_inverse):
    Sigma_posterior_inverse = np.dot(np.transpose(X), X) / (epsilon_sigma**2) + Sigma_prior_inverse
    Sigma_posterior = np.linalg.inv(Sigma_posterior_inverse)
    mu_posterior = np.dot(np.dot(Sigma_posterior, np.transpose(X)), T) / (epsilon_sigma**2)
    mu_posterior = np.transpose(mu_posterior)[0,:]
    return mu_posterior, Sigma_posterior

def question_9(i):
    mu_prior = np.zeros(2)
    Sigma_prior = np.eye(2)
    Sigma_prior_inverse = np.linalg.inv(Sigma_prior)
    if i is 1:
        plot_2d_gaussian(mu_prior, Sigma_prior, filename="Q9-1", save=False)
    elif i is 2 or 3 or 4:
        X, T = generate_data_point()
        mu_posterior, Sigma_posterior = get_posterior(X, T, Sigma_prior_inverse)
        plot_2d_gaussian(mu_posterior, Sigma_posterior, filename="Q9-2", save=False)
        for n in range(6):
            x, t = generate_data_point()
            X = np.vstack([X, x])
            T = np.vstack([T, t])
            mu_posterior, Sigma_posterior = get_posterior(X, T, Sigma_prior_inverse)
            plot_2d_gaussian(mu_posterior, Sigma_posterior, filename="Q9-3-"+str(n+2), save=False)


if __name__ == "__main__":
    question_9(3)