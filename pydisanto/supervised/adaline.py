import numpy as np
from numpy.random import seed


class AdalineGD(object):
    """ADAptive Linear  NEuron Classifier.

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
    cost_: list
        Number of errors cost in each iterations
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
        self.cost_ = []

        for _ in range(self.niterations):
            output = self.net_input(X)
            errors = (y - output)
            self.weights_[1:] += self.learning_rate * X.T.dot(errors)
            self.weights_[0] += self.learning_rate * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def activation(self, X):
        """Computer linear activation"""
        return self.net_input(X)

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.weights_[1:]) + self.weights_[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)


class AdalineSGD(object):
    """ADAptive Linear  NEuron Classifier.

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
    cost_: list
        Number of errors cost in each iterations
    shuffle: bool (default: True)
        Shuffles training data every epoch
    random_state: int (default: None)
        Set random state form shuffling and initializing the weights.
    """

    def __init__(self, learning_rate=0.01, niterations=10, shuffle=True,
                 random_state=None):
        self.learning_rate = learning_rate
        self.niterations = niterations
        self.w_initialized = False
        self.shuffle = shuffle
        if random_state:
            seed(random_state)

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
        self._initialize_weights(X.shape[1])
        self.cost_ = []

        for i in range(self.niterations):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost)/len(y)
            self.cost_.append(avg_cost)
        return self

    def partial_fit(self, X, y):
        """Fit training data without reinitializing the weights"""
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(self, X, y):
        """Shuffle training data"""
        r = np.random.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        """Initialize weights to zero"""
        self.weights_ = np.zeros(1 + m)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        """Apply Adaline learning rule to update the weights"""
        output = self.net_input(xi)
        error = (target - output)
        self.weights_[1:] += self.learning_rate * xi.dot(error)
        self.weights_[0] += self.learning_rate * error
        cost = 0.5 * error**2
        return cost

    def activation(self, X):
        """Computer linear activation"""
        return self.net_input(X)

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.weights_[1:]) + self.weights_[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)
