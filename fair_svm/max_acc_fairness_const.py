from cvxopt import matrix, spdiag, solvers, log
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def load_data():
    data = pd.read_csv('../data/crime/crime_clean.tsv', sep = '\t')
    data = shuffle(data)
    X = data.drop(['crime', 'black'], axis = 1)
    X = X.as_matrix()
    s = np.asarray(data['black'].tolist())
    y = np.asarray(data['crime'].tolist())
    return X, s, y


def logistic_regression(c):
    # X is an n*dim matrix, whose rows are sample points
    X, s, y = load_data()
    n, dim = X.shape
    y = y.reshape(n, 1)
    s = s.reshape(n, 1) - np.mean(s)*np.ones((n, 1))
    G_1 = matrix((1/n)*s.T.dot(X))
    G_2 = matrix((-1/n)*s.T.dot(X))
    G = matrix(np.concatenate((G_1, G_2), axis=0))
    c = matrix(c, (2, 1))

    def log_likelihood(theta=None, z=None):
        if theta is None: return 0, matrix(1.0, (dim, 1))
        theta = theta / 100
        p = sigmoid(X.dot(theta))
        ones = np.ones((n, 1))
        f = matrix(-np.sum(y*np.log(p)+(ones-y)*np.log(ones-p), axis=0))

        # using numpy broadcasting, each row of X is multiplied by (y-p),
        # the sum is taken over the rows to produce the transpose of the gradient
        Df = matrix(-np.sum(X*(y-p), axis=0), (1, dim))

        if z is None: return f, Df
        hessian = X.T.dot((ones-p)*X)
        H = z[0]*hessian
        return f, Df, H
    return solvers.cp(log_likelihood, G=G, h=c)['x']


def acent(A, b):
    m, n = A.size
    def F(x=None, z=None):
        if x is None: return 0, matrix(1.0, (n,1))
        if min(x) <= 0.0: return None
        f = -sum(log(x))
        Df = -(x**-1).T
        if z is None: return f, Df
        H = spdiag(z[0] * x**-2)
        return f, Df, H
    return solvers.cp(F, G=A, h=b)['x']

if __name__=="__main__":
    logistic_regression(1000000000.0)