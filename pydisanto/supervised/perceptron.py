import numpy as np

class Perceptron(object):
    """Perceptron classifier.

    Parameters
    ----------
    learning_rate: float
        Learning rate (between 0.0 and 1.0)
    niterations: int
        Number of iterations over the training set

    Attributes
    ----------
    weights_: 1-d array
        Weights after fitting
    errors_: list
        Number of erros in each iterations
    """

    def __init__(self, learning_rate=0.01, niterations=10):
        self.learning_rate = learning_rate
        self.niterations = niterations

    def fit(self, X, y):
        """Fit the training data.

        Parameters
        ----------
        X: {array-like}, shape = [number of instances, number of features]
            Training matrix
        y: array-like, shape = [number of instances]
            Target values

        Returns
        -------
        self: object
        """
        self.weights_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.niterations):
            errors = 0
            for xi, target in zip(X, y):
                update = self.learning_rate * (target - self.predict(xi))
                self.weights_[1:] += update * xi
                self.weights_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.weights_[1:]) + self.weights_[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)
