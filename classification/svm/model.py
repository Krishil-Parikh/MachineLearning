import numpy as np

class SVM:
    """
    Linear Support Vector Machine (SVM) implemented from scratch.
    """

    def __init__(self, lr=0.001, lambda_param=0.01, epochs=1000, tol=1e-5):
        self.lr = lr
        self.lambda_param = lambda_param
        self.epochs = epochs
        self.tol = tol

        self.w = None
        self.b = None
        self.loss_history = []

    # ===================== CORE ===================== #

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        prev_loss = float("inf")

        for _ in range(self.epochs):
            for idx in range(n_samples):
                condition = y[idx] * (np.dot(X[idx], self.w) - self.b) >= 1

                if condition:
                    dw = 2 * self.lambda_param * self.w
                    db = 0
                else:
                    dw = 2 * self.lambda_param * self.w - y[idx] * X[idx]
                    db = -y[idx]

                self.w -= self.lr * dw
                self.b -= self.lr * db

            loss = self._compute_loss(X, y)
            self.loss_history.append(loss)

            if abs(prev_loss - loss) < self.tol:
                break

            prev_loss = loss

    def predict(self, X):
        X = np.asarray(X)
        decision = np.dot(X, self.w) - self.b
        return np.sign(decision)

    def decision_function(self, X):
        return np.dot(X, self.w) - self.b

    # ===================== LOSS ===================== #

    def _compute_loss(self, X, y):
        distances = 1 - y * (np.dot(X, self.w) - self.b)
        hinge_loss = np.maximum(0, distances)
        return self.lambda_param * np.dot(self.w, self.w) + np.mean(hinge_loss)

    # ===================== METRICS ===================== #

    def accuracy(self, y_true, y_pred):
        return np.mean(y_true == y_pred)