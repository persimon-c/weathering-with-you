# Gaussian Naive Bayes classifier, written from scratch with NumPy.
#
# Assumes all features are continuous and models each one as a Gaussian
# within each class.  Uses log-space arithmetic throughout to avoid
# underflow, and a variance floor to avoid division by zero.
#
# Interface:
#   fit(X, y)          -> self
#   predict_proba(X)   -> (n_samples, n_classes) probabilities
#   predict(X)         -> (n_samples,) predicted labels

import numpy as np


class GaussianNaiveBayes:
    """Gaussian Naive Bayes for binary or multiclass classification."""

    def __init__(self, var_floor=1e-9):
        """
        Parameters
        ----------
        var_floor : float
            Minimum variance for any feature in any class.  Prevents
            division by zero when a feature is constant within a class.
        """
        self.var_floor = var_floor

        # Learned parameters, set by fit()
        self.classes_ = None       # (n_classes,)
        self.log_priors_ = None    # (n_classes,)
        self.means_ = None         # (n_classes, n_features)
        self.variances_ = None     # (n_classes, n_features)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X, y):
        """
        Compute per-class mean, variance, and prior from the training set.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : ndarray of shape (n_samples,)  — integer class labels

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        self.means_ = np.zeros((n_classes, n_features))
        self.variances_ = np.zeros((n_classes, n_features))
        self.log_priors_ = np.zeros(n_classes)

        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.means_[i] = X_c.mean(axis=0)
            self.variances_[i] = X_c.var(axis=0)
            self.log_priors_[i] = np.log(X_c.shape[0] / X.shape[0])

        # Apply variance floor
        self.variances_ = np.maximum(self.variances_, self.var_floor)

        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_proba(self, X):
        """
        Posterior probabilities for each class via Bayes rule.

        Computes the joint log-likelihood for each class using the
        Gaussian PDF in log space, adds the log prior, then normalises
        with log-sum-exp to get calibrated probabilities.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
        """
        X = np.asarray(X, dtype=np.float64)

        # X[:, None, :] is (n, 1, d), means_[None, :, :] is (1, C, d)
        diff = X[:, None, :] - self.means_[None, :, :]

        # Log-likelihood per feature per class: Gaussian log-PDF
        # -0.5 * log(2 * pi * var) - (x - mu)^2 / (2 * var)
        log_likelihood = (
            -0.5 * np.log(2.0 * np.pi * self.variances_[None, :, :])
            - (diff ** 2) / (2.0 * self.variances_[None, :, :])
        )

        # Sum over features, add log prior  ->  (n_samples, n_classes)
        joint = log_likelihood.sum(axis=2) + self.log_priors_[None, :]

        # Stable log-sum-exp normalisation
        max_joint = joint.max(axis=1, keepdims=True)
        log_norm = max_joint + np.log(
            np.exp(joint - max_joint).sum(axis=1, keepdims=True)
        )
        log_proba = joint - log_norm

        return np.exp(log_proba)

    def predict(self, X):
        """
        Predict class labels (argmax of posterior probabilities).

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

