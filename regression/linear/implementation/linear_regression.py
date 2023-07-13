import numpy as np


class LinearReg(object):
    def __init__(self, indim=1, outdim=1):
        # initialize the parameters first. 
        self.W = np.zeros(shape=(indim+1, outdim))
    
    def fit(self, X, T):
        # implement the .fit() using the simple least-square closed-form solution:
        X = np.hstack([X, np.ones(shape=[X.shape[0], 1])])
        self.W = np.linalg.inv(X.T@X) @ X.T @ T
        # HINT:
        #   extend the input features before fitting to it.
        #   compute the weight matrix of shape [indim+1, outdim]

    def predict(self, X):
        # implement the .predict() using the parameters learned by .fit()
        X = np.hstack([X, np.ones(shape=[X.shape[0], 1])])
        return X @ self.W


def second_order_basis(X):
    # we will perform a simple implementation
    # using the broadcasting mechanism in numpy.
    # HINT:
    #   np.triu_indices(): returns the indices of the upper triangular matrix
    res, dim = [], X.shape[1]
    for x in X:
        x = np.expand_dims(x, 1)
        x_basis = x * x.T
        res.append(x_basis[np.tril_indices(dim)])

    return np.vstack(res)
