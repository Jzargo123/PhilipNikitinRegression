import numpy as np
import pickle
import os
from scipy import optimize
from abc import abstractmethod


class Regression:
    """
        Abstract class for regression models
        Parameters
        ----------
        intercept : boolean, optional, default True
            whether to calculate the intercept for this model. If set
            to False, no intercept will be used in calculations

        reg_l1 : float, optional, default 0
            coefficient of L1 regularisation

        reg_l2 : float, optional, default 0
            coefficient of L2 regularisation

        optimiser: strng, optional, default 'L-BFGS-B'
            method of optimisation: BFGS, 'L-BFGS-B' is supported. May be some others methods
            from scipy.optimize.minimise is worked but it isn't tested

        Examples
        --------
        >>> import numpy as np
        >>> from sklearn.linear_model import LinearRegression
        >>> X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
        >>> # y = 1 * x_0 + 2 * x_1 + 3
        >>> y = np.dot(X, np.array([1, 2])) + 3
        >>> reg = LinearRegression().fit(X, y)
        >>> reg.predict(np.array([[3, 5]]))
        array([16.])
        """

    def __init__(self, reg_l1=0.5, reg_l2=0.5, optimizator='L-BFGS-B', intercept=True):
        self.reg_l1 = reg_l1
        self.reg_l2 = reg_l2
        self.weights = None
        self.optimizator = optimizator
        self.intercept = intercept

    @abstractmethod
    def compute_loss(self, weights, features, labels):
        """
        Abstract methods for computing loss

        :param weights: numpy.ndarray: weights of the model
        :param labels: numpy.ndarray: labels
        :param features: numpy.ndarray: features
        :return:
        """
        pass

    @abstractmethod
    def compute_grad(self, weights, features, labels):
        """
        Abstract methods for computing gradient

        :param weights: numpy.ndarray: weights of the model
        :param labels: numpy.ndarray: labels
        :param features: numpy.ndarray: features
        :return:
        """
        pass

    def fit_intercept(self, features):
        """
        Adds intercept according to model parametrs

        :param features: numpy.ndarray: features
        :return: features: with intercept
        """
        if self.intercept:
            return np.c_[features, np.ones(features.shape[0])]
        return features

    def fit(self, features, labels):
        """
        Fits the model with the features and labels

        :param labels: numpy.ndarray: labels
        :param features: numpy.ndarray: features
        :return:
        """
        if len(labels) == 0:
            raise ValueError("More then zero exampes are expected")
        features = self.fit_intercept(features)
        init_weights = np.random.rand(features.shape[1]) * 1e-2
        if np.sum(np.abs(features)) > 1e-16:
            self.weights = optimize.minimize(x0=init_weights,
                                             fun=lambda w: self.compute_loss(w, features, labels),
                                             jac=lambda w: self.compute_grad(w, features, labels),
                                             method=self.optimizator, tol=1e-10)["x"]
        else:
            self.weights = init_weights

    def predict(self, features):
        """
        Predicts labels

        :param features: numpy.ndarray: features
        :return: numpy.ndarray: predicted labels
        """
        features = self.fit_intercept(features)
        if self.weights is not None:
            y_estimate = features.dot(self.weights)
            return y_estimate
        else:
            raise ValueError('You must fit of load the model before predict')

    def save_model(self, weights_path):
        """
        Saves the model

        :param weights_path: str: path to the model file
        :return:
        """
        with open(weights_path, 'wb') as f:
            pickle.dump(self.weights, f)

    def load_model(self, weights_path):
        """
        Loads the model

        :param weights_path: str: path to the model file
        :return:
        """
        if os.path.exists(weights_path):
            with open(weights_path, 'rb') as f:
                self.weights = pickle.load(f)
        else:
            raise ValueError('File {} does not exist'.format(weights_path))


class LinearRegression(Regression):
    """
    Ordinary least squares Linear Regression.
    """
    def compute_loss(self, weights, features, labels):
        """
        Computes mean squred loss

        :param weights: numpy.ndarray: weights of the model
        :param labels: numpy.ndarray: labels
        :param features: numpy.ndarray: features
        :return: float: loss
        """
        mse_loss = np.sum((features.dot(weights) - labels) ** 2) / len(features)
        reg_l1_loss = self.reg_l1 * np.sum(np.abs(weights))
        reg_l2_loss = self.reg_l2 * weights.dot(weights)
        return mse_loss + reg_l1_loss + reg_l2_loss

    def compute_grad(self, weights, features, labels):
        """
        Computes mean squred gradient

        :param weights: numpy.ndarray: weights of the model
        :param labels: numpy.ndarray: labels
        :param features: numpy.ndarray: features
        :return: numpy.ndarray: gradient
        """
        mse_grad = 2 * (features.dot(weights) - labels).dot(features) / len(features)
        reg_l1_grad = self.reg_l1 * np.sign(weights)
        reg_l2_grad = self.reg_l2 * 2 * weights
        return mse_grad + reg_l1_grad + reg_l2_grad
