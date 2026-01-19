import numpy as np

class HuberRegression:
    """
    Huber Regression implemented from scratch using NumPy.

    Robust to outliers by combining:
    - L2 loss for small errors
    - L1 loss for large errors
    """

    def __init__(self, lr=0.01, epochs=1000, delta=1.35, tolerance=1e-5):
        self.lr = lr
        self.epochs = epochs
        self.delta = delta
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

    # ===================== CORE ===================== #

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

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

            gradient = np.zeros_like(self.weights)

            for i in range(n_samples):
                if abs(errors[i]) <= self.delta:
                    grad = errors[i]
                else:
                    grad = self.delta * np.sign(errors[i])

                gradient += grad * X[i]

            gradient /= n_samples
            self.weights -= self.lr * gradient

            loss = self._huber_loss(errors)
            self.loss_history.append(loss)

            if abs(prev_loss - loss) < self.tolerance:
                break

            prev_loss = loss

    def predict(self, X):
        X = np.asarray(X)
        X = self._normalize(X)
        X = self._add_bias(X)
        return X @ self.weights

    # ===================== LOSSES ===================== #

    def _huber_loss(self, errors):
        loss = 0.0
        for e in errors:
            if abs(e) <= self.delta:
                loss += 0.5 * e ** 2
            else:
                loss += self.delta * (abs(e) - 0.5 * self.delta)
        return loss / len(errors)

    # ===================== METRICS ===================== #

    def mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def r2_score(self, y_true, y_pred):
        ss_total = np.sum((y_true - y_true.mean()) ** 2)
        ss_res = np.sum((y_true - y_pred) ** 2)
        return 1 - ss_res / ss_total