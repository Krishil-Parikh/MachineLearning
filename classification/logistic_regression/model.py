import numpy as np

class LogisticRegression:
    """
    Binary Logistic Regression implemented from scratch using NumPy.
    """

    def __init__(self, lr=0.01, epochs=1000, tolerance=1e-6):
        self.lr = lr
        self.epochs = epochs
        self.tolerance = tolerance

        self.weights = None
        self.means = None
        self.stds = None
        self.loss_history = []

    # ===================== HELPERS ===================== #

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _add_bias(self, X):
        return np.c_[np.ones(X.shape[0]), X]

    def _normalize(self, X):
        safe_stds = np.where(self.stds == 0, 1, self.stds)
        return (X - self.means) / safe_stds

    # ===================== CORE METHODS ===================== #

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
            linear_output = X @ self.weights
            y_pred = self._sigmoid(linear_output)

            errors = y_pred - y
            gradient = (1 / n_samples) * (X.T @ errors)

            self.weights -= self.lr * gradient

            loss = self._log_loss(y, y_pred)
            self.loss_history.append(loss)

            if abs(prev_loss - loss) < self.tolerance:
                break

            prev_loss = loss

    def predict_proba(self, X):
        X = np.asarray(X)
        X = self._normalize(X)
        X = self._add_bias(X)
        return self._sigmoid(X @ self.weights)

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    # ===================== LOSSES & METRICS ===================== #

    def _log_loss(self, y_true, y_pred):
        eps = 1e-9
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        )

    def accuracy(self, y_true, y_pred):
        return np.mean(y_true == y_pred)