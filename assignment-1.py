import matplotlib.pyplot as plt
import tikzplotlib as tikz
import numpy as np
import pylab as pb
import random
from math import pi, sin
from numpy.linalg import inv
from scipy.spatial.distance import cdist
from scipy.stats import multivariate_normal as mvn
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Matern

from gaussian_processes_util import plot_gp

random.seed(1000)
np.random.seed(1000)

W = np.array([[0.5], [-1.5]])
X_Set = np.linspace(-1.0, 1.0, num=201)
# epsilon_sigma_list = [0.1, 0.2, 0.4, 0.8]

X_Set = np.array([-4, -3, -2, -1, 0, 2, 3, 5])
epsilon_sigma = 0.3
l_list = [0.1, 1, 10, 100]


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
        tikz.save("./figure/"+filename+".tex")
    else:
        plt.show()

def plot_function(mu, Sigma, X, T, sample=10, filename="linear_function", save=True):
    X_Set_Ones = np.ones_like(X_Set)
    X_Set_Expanded = np.transpose(np.vstack([X_Set, X_Set_Ones]))
    
    _, ax = plt.subplots()
    for _ in range(sample):
        W = np.random.multivariate_normal(mu, Sigma)
        T_Set = np.dot(X_Set_Expanded, W)
        plt.plot(X_Set, T_Set)

    plt.scatter(X, T, c="blue")
    ax.set_aspect('equal', 'box')

    if save:
        tikz.save("./figure/"+filename+".tex")
    else:
        plt.show()

def linear_function(W_Model, x, epsilon=0.0):
    return np.dot(x, W_Model) + epsilon

def generate_data_point(epsilon_sigma):
    epsilon = np.array([[np.random.normal(0.0, epsilon_sigma)]])
    x = np.array([[random.choice(X_Set), 1.0]])
    t = linear_function(W, x, epsilon)
    return x, t

def get_posterior(X, T, Sigma_prior_inverse, epsilon_sigma):
    Sigma_posterior_inverse = np.dot(np.transpose(X), X) / (epsilon_sigma**2) + Sigma_prior_inverse
    Sigma_posterior = inv(Sigma_posterior_inverse)
    mu_posterior = np.dot(np.dot(Sigma_posterior, np.transpose(X)), T) / (epsilon_sigma**2)
    mu_posterior = np.transpose(mu_posterior)[0,:]
    return mu_posterior, Sigma_posterior

def question_9(i, epsilon_sigma, save):
    mu_prior = np.zeros(2)
    Sigma_prior = np.eye(2)
    Sigma_prior_inverse = inv(Sigma_prior)
    if i is 1:
        plot_2d_gaussian(mu_prior, Sigma_prior,
                        filename="Q9-1", save=save)
    elif i is 2 or 3 or 4:
        X, T = generate_data_point(epsilon_sigma)
        mu_posterior, Sigma_posterior = get_posterior(X, T, Sigma_prior_inverse, epsilon_sigma)
        plot_2d_gaussian(mu_posterior, Sigma_posterior, 
                        filename=str(epsilon_sigma)+"-Q9-2", save=save)
        plot_function(mu_posterior, Sigma_posterior, X[:,0], T, sample=5, 
                        filename=str(epsilon_sigma)+"-Q9-3-F", save=save)
        if i is 4:
            for n in range(6):
                x, t = generate_data_point(epsilon_sigma)
                X = np.vstack([X, x])
                T = np.vstack([T, t])
                mu_posterior, Sigma_posterior = get_posterior(X, T, Sigma_prior_inverse, epsilon_sigma)
                plot_2d_gaussian(mu_posterior, Sigma_posterior, 
                                filename=str(epsilon_sigma)+"-Q9-4-"+str(n+2), save=save)
                plot_function(mu_posterior, Sigma_posterior, X[:,0], T, sample=5, 
                                filename=str(epsilon_sigma)+"-Q9-4-"+str(n+2)+"-F", save=save)
        
        print(np.transpose(np.vstack([X[:,0], T[:,0]])))

###########################################################################
def plot_GP_prior(sigma_f=1.0, l=1.0, filename="GP_prior", save=True):
    num = 200
    X = np.linspace(-10.0, 10.0, num=num).reshape(-1, 1)
    mu = np.zeros(X.shape)
    K = kernel(X, X, sigma_f=sigma_f, l=l)
    samples = np.random.multivariate_normal(mu.ravel(), K, 10)
    plot_gp(mu, K, X, samples=samples)

def nonlinear_function(x, epsilon=0.0):
    # T_train = (2*np.ones_like(X_train) + (0.5*X_train-np.ones_like(X_train))**2) * np.sin(3*X_train) + np.random.normal(0, 0.3, len(X_Set)).reshape(-1 ,1)
    return (2 + (0.5 * x -1)**2) * sin(3 * x) + epsilon

def kernel(X1, X2, sigma_f=1.0, l=1.0):
    """
    Squared exponential kernel. Computes a covariance matrix.
    Args:
        X1: Array of m points (m x d)
        X2: Array of n points (n x d)
    Returns:
        Covariance matrix (m x n)
    """
    sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    return sigma_f**2 * np.exp(-0.5 / l**2 * sqdist)

def posterior_predictive(X_s, X_train, Y_train, l=1.0, sigma_f=1.0, sigma_y=1e-8):
    '''
    From krasserm
    Computes the suffifient statistics of the GP posterior predictive distribution 
    from m training data X_train and Y_train and n new inputs X_s.

    Args:
        X_s: New input locations (n x d).
        X_train: Training locations (m x d).
        Y_train: Training targets (m x 1).
        l: Kernel length parameter.
        sigma_f: Kernel vertical variation parameter.
        sigma_y: Noise parameter.

    Returns:
        Posterior mean vector (n x d) and covariance matrix (n x n).
    '''
    K = kernel(X_train, X_train, l, sigma_f) + sigma_y**2 * np.eye(len(X_train))
    K_s = kernel(X_train, X_s, l, sigma_f)
    K_ss = kernel(X_s, X_s, l, sigma_f) + sigma_y**2 * np.eye(len(X_s))
    K_inv = inv(K)
    # Equation (4)
    mu_s = K_s.T.dot(K_inv).dot(Y_train)
    # Equation (5)
    cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)
    return mu_s, cov_s

def question_10(save):
    # plot_GP_prior(l=1)

    num = 200
    X_s = np.linspace(-10.0, 10.0, num=num).reshape(-1, 1)
    X_train = X_Set.reshape(-1, 1)
    T_train = (2*np.ones_like(X_train) + (0.5*X_train-np.ones_like(X_train))**2) * np.sin(3*X_train) + np.random.normal(0, 0.3, len(X_Set)).reshape(-1 ,1)
    mu_s, cov_s = posterior_predictive(X_s, X_train, T_train, sigma_f=1.0, l=1.0, sigma_y=1.0)
    samples = np.random.multivariate_normal(mu_s.ravel(), cov_s, 3)
    plot_gp(mu_s, cov_s, X_s, X_train=X_train, Y_train=T_train, samples=samples)

if __name__ == "__main__":
    # for epsilon_sigma in epsilon_sigma_list:
        # question_9(4, epsilon_sigma, save=True)
    
    question_10(save=False)