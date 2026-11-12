# Logistic regression with L2 regularisation and class weighting,
# written from scratch with NumPy.
#
# Uses gradient descent on binary cross-entropy loss.  The class-weight
# option up-weights the minority class (flooded, ~17 %) so that the
# gradient is not dominated by the majority.
#
# Interface:
#   fit(X, y)          -> self
#   predict_proba(X)   -> (n_samples, 2) probabilities [P(0), P(1)]
#   predict(X)         -> (n_samples,) predicted labels

import numpy as np


class LogisticRegression:
    """Binary logistic regression with L2 and optional class weights."""

    def __init__(self, lr=0.01, epochs=1000, reg=0.01,
                 class_weight=None, threshold=0.5, seed=42):
        """
        Parameters
        ----------
        lr : float
            Learning rate for gradient descent.
        epochs : int
            Number of full-batch gradient descent steps.
        reg : float
            L2 regularisation strength (lambda).  Applied to weights
            only, not the bias.
        class_weight : str or None
            If ``"balanced"``, each sample is weighted inversely
            proportional to its class frequency.  Otherwise uniform.
        threshold : float
            Decision threshold for ``predict()``.
        seed : int
            Random seed for weight initialisation.
        """
        self.lr = lr
        self.epochs = epochs
        self.reg = reg
        self.class_weight = class_weight
        self.threshold = threshold
        self.seed = seed

        # Learned parameters, set by fit()
        self.w_ = None             # (n_features,)
        self.b_ = None             # scalar
        self.loss_history_ = None  # (epochs,)

    # ------------------------------------------------------------------
    # Numerically stable sigmoid
    # ------------------------------------------------------------------

    @staticmethod
    def _sigmoid(z):
        """Piecewise sigmoid that avoids overflow in exp."""
        out = np.empty_like(z, dtype=np.float64)
        pos = z >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        exp_z = np.exp(z[~pos])
        out[~pos] = exp_z / (1.0 + exp_z)
        return out

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X, y):
        """
        Learn weights by gradient descent on (weighted) cross-entropy
        with L2 regularisation.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Should already be standardised.
        y : ndarray of shape (n_samples,)  — 0 or 1

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n, d = X.shape

        # Initialise weights near zero
        rng = np.random.default_rng(self.seed)
        self.w_ = rng.normal(0, 0.01, size=d)
        self.b_ = 0.0

        # Per-sample weights for class imbalance
        sample_w = np.ones(n, dtype=np.float64)

        if self.class_weight == "balanced":
            classes, counts = np.unique(y, return_counts=True)
            weight_map = {c: n / (len(classes) * cnt)
                          for c, cnt in zip(classes, counts)}
            sample_w = np.array([weight_map[yi] for yi in y])

        self.loss_history_ = np.zeros(self.epochs)
        eps = 1e-15  # for log stability

        for epoch in range(self.epochs):
            # Forward pass
            z = X @ self.w_ + self.b_
            p = self._sigmoid(z)

            # Weighted cross-entropy loss + L2
            raw_loss = -(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
            data_loss = np.mean(sample_w * raw_loss)
            reg_loss = 0.5 * self.reg * np.dot(self.w_, self.w_)
            self.loss_history_[epoch] = data_loss + reg_loss

            # Gradients (with sample weighting)
            error = (p - y) * sample_w
            grad_w = (X.T @ error) / n + self.reg * self.w_
            grad_b = error.mean()

            # Update
            self.w_ -= self.lr * grad_w
            self.b_ -= self.lr * grad_b

        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_proba(self, X):
        """
        Class probabilities [P(class=0), P(class=1)] per sample.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        proba : ndarray of shape (n_samples, 2)
        """
        X = np.asarray(X, dtype=np.float64)
        p1 = self._sigmoid(X @ self.w_ + self.b_)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        """
        Predict class labels using the decision threshold.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] >= self.threshold).astype(int)

