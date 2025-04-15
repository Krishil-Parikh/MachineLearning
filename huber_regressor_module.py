import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class LinearRegression:
    def __init__(self, alpha=0.01, epochs=1000 , epsilom = 1.35):
        self.alpha = alpha  # Learning rate
        self.epochs = epochs  # Number of iterations for gradient descent
        self.m = None  # Coefficients
        self.means = None  # Means for normalization
        self.stds = None  # Standard deviations for normalization
        self.cost_history = []
        self.epsilom = epsilom

    def fit(self, X, y):
        # Convert y to a NumPy array to avoid indexing issues
        y = np.array(y)
        
        # Normalize the features
        self.means = X.mean(axis=0)
        self.stds = X.std(axis=0)
        X = (X - self.means) / self.stds

        # Add a column of ones to X for the intercept (bias) term
        X = np.c_[np.ones(X.shape[0]), X]

        # Initialize parameters
        self.m = np.zeros(X.shape[1])

        # Gradient descent
        for epoch in range(self.epochs):
            self.m = self.huber_loss_multi(X, y)
            cost = np.mean((np.dot(X, self.m) - y) ** 2)
            self.cost_history.append(cost)

            # if epoch % 100 == 0:
            #     print(f"Epoch {epoch}: Cost = {cost}, Coefficients = {self.m}")

    def plot(self):
        # Plot cost history
        plt.figure(figsize=(14, 6))
        plt.plot(range(self.epochs), self.cost_history, label='Cost Function Convergence')
        plt.xlabel('Epochs')
        plt.ylabel('Cost (MSE)')
        plt.title('Convergence of Gradient Descent for Multiple Linear Regression')
        plt.legend()
        plt.show()

    def gradient_descent_multi(self, X, y):
        m_gradient = np.zeros_like(self.m)
        n = len(y)

        for i in range(n):
            y_pred = np.dot(X[i], self.m)
            error = y_pred - y[i]
            m_gradient += (1/n) * error * X[i]

        return self.m - self.alpha * m_gradient
    
    def huber_loss_multi(self, X, y):
        m_gradient = np.zeros_like(self.m)
        n = len(y)

        for i in range(n):
            y_pred = np.dot(X[i], self.m)
            error = y_pred - y[i]
            if abs(error) <= self.epsilom:
                m_gradient += (1/n) * error * X[i]
                # print("No Huber Loss")
            else:
                m_gradient += (1/n) * self.epsilom * error * X[i]
                # print("Huber Loss")
        
        return self.m - self.alpha * m_gradient
    
    def compute_huber_cost(self, X, y):
        cost = 0
        n = len(y)
        for i in range(n):
            y_pred = np.dot(X[i], self.m)
            error = y_pred - y[i]
            if abs(error) <= self.epsilom:
                cost += (1/n) * 0.5 * error ** 2
            else:
                cost += (1/n) * self.epsilom * (abs(error) - 0.5 * self.epsilom)
        return cost


    def predict(self, new_data):
        new_data = np.array(new_data)
        new_data_normalized = (new_data - self.means) / self.stds
        new_data_normalized = np.c_[np.ones(new_data_normalized.shape[0]), new_data_normalized]
        predictions = np.dot(new_data_normalized, self.m)
        return predictions

    def mean_squared_error(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    def accuracy(self, y_true, y_pred):
        return 1 - (np.mean(np.abs(y_true - y_pred)) / np.mean(y_true))

