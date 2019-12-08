import itertools as it
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

from math import exp, pi, sqrt


def plotDataSet(data):
    _, ax = plt.subplots()
    plt.xlim((-1.5, 1.5))
    plt.ylim((-1.5, 1.5))
    ax.set_aspect('equal')
    ax.axis('off')
    for i in range(3):      # x1
        for j in range(3):  # x2
            if data[i,j] > 0:
                ax.plot(i-1, j-1, 'bo', markersize=40, mfc='none')
            else:
                ax.plot(i-1, j-1, 'rx', markersize=40)


def generateDataSets():
    """
    Generate all possible data sets.
    Each data set is a numpy.array.
    """
    DList = list(it.product([-1, 1], repeat=9))
    datasets = []
    for D in DList:
        datasets.append(np.array(D).reshape(3,3))
    return datasets


def samplePrior(model, S, sigma=1000):
    """
    Sample from prior distribution of theta.
    Samples are in a single numpy.array([]).
    Each sample is in a row.
    """
    if model is 0:
        return np.zeros(S)
    # mu = np.zeros(model)
    mu = 5.0 * np.ones(model)
    Sigma = sigma * np.eye(model)
    return np.random.multivariate_normal(mu, Sigma, S)


def getLikelihood(data, model, theta):
    if model is 0:
        return 1 / 512

    p = 1.0
    for i in range(3):      # x1
        for j in range(3):  # x2
            if model is 1:
                p *= 1 / (1 + exp(-data[i,j] * theta[0] * (i-1)))
            elif model is 2:
                p *= 1 / (1 + exp(-data[i,j] * (theta[0] * (i-1) + theta[1] * (j-1))))
            elif model is 3:
                p *= 1 / (1 + exp(-data[i,j] * (theta[0] * (i-1) + theta[1] * (j-1) + theta[2])))
    return p


def getEvidence(model, S=10**8):
    np.random.seed(1000)

    DList = generateDataSets()
    thetaList = samplePrior(model, S)
    evidence = []
    for data in DList:
        p = 0.0
        for theta in thetaList:
            p += getLikelihood(data, model, theta)
        p /= S
        evidence.append(p)

    return evidence


if __name__ == "__main__":
    fig, ax = plt.subplots()
    evidenceList = []
    for model in range(4):
        if model is 0:
            style = '--m'
            label = '$P(\mathcal{D} | M_{0})$'
        elif model is 1:
            style = 'b'
            label = '$P(\mathcal{D} | M_{1})$'
        elif model is 2:
            style = 'r'
            label = '$P(\mathcal{D} | M_{2})$'
        elif model is 3:
            style = 'g'
            label = '$P(\mathcal{D} | M_{3})$'
        evidence = getEvidence(model, S=10000)
        evidenceList.append(evidence)
        ax.plot(range(len(evidence)), evidence, style, linewidth=1,label=label)
    ax.legend(loc='upper right')
    plt.ylabel('Evidence')
    plt.xlabel('All data sets, $\mathcal{D}$')
    plt.show()