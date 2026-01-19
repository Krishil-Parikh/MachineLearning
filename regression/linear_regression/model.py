import numpy as np

class LinearRegression:
    """
    Linear Regression implemented from scratch using NumPy.

    Supports:
    - Feature normalization
    - Full-batch & mini-batch gradient descent
    - L2 regularization (Ridge)
    - Early stopping
    """

    def __init__(
        self,
        lr=0.01,
        epochs=1000,
        tolerance=1e-5,
        l2_lambda=0.0,
        batch_size=None
    ):
        self.lr = lr
        self.epochs = epochs
        self.tolerance = tolerance
        self.l2_lambda = l2_lambda
        self.batch_size = batch_size

        self.weights = None
        self.means = None
        self.stds = None
        self.loss_history = []

    # ===================== INTERNAL HELPERS ===================== #

    def _add_bias(self, X):
        return np.c_[np.ones(X.shape[0]), X]

    def _normalize(self, X):
        safe_stds = np.where(self.stds == 0, 1, self.stds)
        return (X - self.means) / safe_stds

    # ===================== CORE API ===================== #

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
        batch_size = n_samples if self.batch_size is None else self.batch_size

        for epoch in range(self.epochs):
            indices = np.random.permutation(n_samples)

            for start in range(0, n_samples, batch_size):
                batch_idx = indices[start:start + batch_size]
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]

                predictions = X_batch @ self.weights
                errors = predictions - y_batch

                gradient = (2 / len(y_batch)) * (X_batch.T @ errors)

                if self.l2_lambda > 0:
                    gradient[1:] += (2 * self.l2_lambda / len(y_batch)) * self.weights[1:]

                self.weights -= self.lr * gradient

            loss = self._compute_loss(X, y)
            self.loss_history.append(loss)

            if abs(prev_loss - loss) < self.tolerance:
                break

            if np.isnan(loss) or np.isinf(loss):
                break

            prev_loss = loss

    def predict(self, X):
        X = np.asarray(X)
        X = self._normalize(X)
        X = self._add_bias(X)
        return X @ self.weights

    # ===================== METRICS ===================== #

    def mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def r2_score(self, y_true, y_pred):
        ss_total = np.sum((y_true - y_true.mean()) ** 2)
        ss_residual = np.sum((y_true - y_pred) ** 2)
        return 1 - ss_residual / ss_total

    # ===================== INTERNAL ===================== #

    def _compute_loss(self, X, y):
        predictions = X @ self.weights
        errors = predictions - y
        loss = np.mean(errors ** 2)

        if self.l2_lambda > 0:
            loss += self.l2_lambda * np.sum(self.weights[1:] ** 2) / len(y)

        return loss
    
    def plot_loss(loss_history):
        import matplotlib.pyplot as plt

        plt.plot(loss_history)
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.title("Training Loss")
        plt.grid(alpha=0.3)
        plt.show()