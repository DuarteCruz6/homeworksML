import numpy as np
from math import sqrt, pi, exp

# Dataset: (x1, x2, Class)
X = np.array([
    [-2, 1, 'A'],
    [-1, 2, 'A'],
    [ 1, 3, 'A'],
    [ 2, 4, 'A'],
    [-1, 0.4, 'B'],
    [ 2, 1.1, 'B'],
    [ 3, 1.5, 'B'],
    [ 4, 1.8, 'B']
], dtype=object)

# Query point
x_query = np.array([1.0, 2.0])
posterior = {}

# Step 1: split data by class
classes = np.unique(X[:,2])

# Gaussian pdf for 1D
def gaussian(x, mu, sigma):
    return (1.0/(sqrt(2*pi)*sigma)) * exp(-((x-mu)**2)/(2*sigma**2))

for c in classes:

    Xc = X[X[:,2] == c][:,:2].astype(float) # exemplos da classe
    prior = len(Xc) / len(X)   # prior = fraction of samples

    mu = Xc.mean(axis=0)           # mean vector per class
    var = Xc.var(axis=0)           # variance vector (MLE)
    sigma = np.sqrt(var)           # std deviation
    likelihood = np.prod([gaussian(x_query[i], mu[i], sigma[i]) for i in range(2)])

    posterior[c] = prior * likelihood

    print("prior for class ", c, " is ", prior)
    print("mu for class ", c, " is ", mu)
    print("sigma for class ", c, " is ", sigma)
    print("likelihood for class ", c, " is ", likelihood)
    print("posterior for class ", c, " is ", posterior[c])

print("Predicted class:", max(posterior, key=posterior.get))