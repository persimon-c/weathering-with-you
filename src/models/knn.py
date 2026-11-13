# K nearest neighbors classifier, written from scratch with NumPy.
#
# Stores the training rows and labels. A new row gets the share of
# flooded rows among its k nearest training rows by Euclidean distance.
# Features should already be standardized so no column dominates.
#
# Interface:
#   fit(X, y)          -> self
#   predict_proba(X)   -> (n_samples, 2) probabilities [P(0), P(1)]
#   predict(X)         -> (n_samples,) predicted labels

import numpy as np


class KNearestNeighbors:
    """Binary k nearest neighbors with a majority vote."""

    def __init__(self, k=5, threshold=0.5):
        self.k = k
        self.threshold = threshold
        self.X_ = None
        self.y_ = None

    def fit(self, X, y):
        self.X_ = np.asarray(X, dtype=np.float64)
        self.y_ = np.asarray(y, dtype=np.float64)
        if not 1 <= self.k <= len(self.y_):
            raise ValueError(f"k={self.k} must be between 1 and {len(self.y_)}")
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        # Squared distances as |a|^2 + |b|^2 - 2ab, one (n_query, n_train) matrix.
        d = (X ** 2).sum(axis=1)[:, None] + (self.X_ ** 2).sum(axis=1)[None, :] - 2.0 * X @ self.X_.T
        # Stable sort keeps the earlier training row on equal distances, so runs repeat.
        nearest = np.argsort(d, axis=1, kind="stable")[:, :self.k]
        p1 = self.y_[nearest].mean(axis=1)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)
