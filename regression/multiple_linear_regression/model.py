import numpy as np

class MultipleLinearRegression:
    """
    Multiple Linear Regression implemented from scratch using NumPy.

    Supports:
    - Feature normalization
    - Batch gradient descent
    - Optional L2 regularization (Ridge)
    """

    def __init__(
        self,
        lr=0.01,
        epochs=1000,
        l2_lambda=0.0,
        tolerance=1e-6
    ):
        self.lr = lr
        self.epochs = epochs
        self.l2_lambda = l2_lambda
        self.tolerance = tolerance

        self.weights = None
        self.means = None
        self.stds = None
        self.loss_history = []

    # ===================== HELPERS ===================== #

    def _add_bias(self, X):
        return np.c_[np.ones(X.shape[0]), X]

    def _normalize(self, X):
        safe_stds = np.where(self.stds == 0, 1, self.stds)
        return (X - self.means) / safe_stds

    # ===================== CORE METHODS ===================== #

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        # Feature normalization
        self.means = X.mean(axis=0)
        self.stds = X.std(axis=0)

        X = self._normalize(X)
        X = self._add_bias(X)

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)

        prev_loss = float("inf")

        for _ in range(self.epochs):
            predictions = X @ self.weights
            errors = predictions - y

            gradient = (2 / n_samples) * (X.T @ errors)

            if self.l2_lambda > 0:
                gradient[1:] += (2 * self.l2_lambda / n_samples) * self.weights[1:]

            self.weights -= self.lr * gradient

            loss = self._mse_loss(errors)
            self.loss_history.append(loss)

            if abs(prev_loss - loss) < self.tolerance:
                break

            prev_loss = loss

    def predict(self, X):
        X = np.asarray(X)
        X = self._normalize(X)
        X = self._add_bias(X)
        return X @ self.weights

    # ===================== LOSSES & METRICS ===================== #

    def _mse_loss(self, errors):
        return np.mean(errors ** 2)

    def mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def r2_score(self, y_true, y_pred):
        ss_total = np.sum((y_true - y_true.mean()) ** 2)
        ss_residual = np.sum((y_true - y_pred) ** 2)
        return 1 - ss_residual / ss_total