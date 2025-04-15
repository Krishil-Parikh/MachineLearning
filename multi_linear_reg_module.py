import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class LinearRegression:
    def __init__(self, alpha=0.01, epochs=1000, tolerance=1e-5, regularization=0, batch_size=None):
        self.alpha = alpha
        self.epochs = epochs
        self.tolerance = tolerance
        self.regularization = regularization
        self.batch_size = batch_size
        self.coefficients = None
        self.means = None
        self.stds = None
        self.cost_history = []

    def _normalize(self, X):
        # Handle zero std case
        safe_stds = np.where(self.stds == 0, 1, self.stds)
        return (X - self.means) / safe_stds

    def _add_intercept(self, X):
        return np.c_[np.ones(X.shape[0]), X]

    def fit(self, X, y):
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Compute normalization parameters
        self.means = np.mean(X, axis=0)
        self.stds = np.std(X, axis=0)
        
        # Prepare data
        X_normalized = self._normalize(X)
        X_normalized = self._add_intercept(X_normalized)
        
        # Initialize model
        self.coefficients = np.zeros(X_normalized.shape[1])
        
        # Constants for optimization
        n = len(y)
        grad_factor = 2/n
        batch_size = n if self.batch_size is None else min(self.batch_size, n)
        
        # Gradient descent
        prev_cost = float('inf')
        for epoch in range(self.epochs):
            if self.batch_size is None:
                # Full-batch gradient descent
                y_pred = np.dot(X_normalized, self.coefficients)
                errors = y_pred - y
                
                gradient = grad_factor * np.dot(X_normalized.T, errors)
                
                # Add regularization if needed
                if self.regularization > 0:
                    gradient[1:] += (2 * self.regularization * self.coefficients[1:]) / n
                
                self.coefficients -= self.alpha * gradient
            else:
                # Mini-batch gradient descent
                indices = np.random.permutation(n)
                for start_idx in range(0, n, batch_size):
                    batch_indices = indices[start_idx:start_idx + batch_size]
                    X_batch = X_normalized[batch_indices]
                    y_batch = y[batch_indices]
                    
                    y_pred = np.dot(X_batch, self.coefficients)
                    errors = y_pred - y_batch
                    
                    batch_grad_factor = 2/len(batch_indices)
                    gradient = batch_grad_factor * np.dot(X_batch.T, errors)
                    
                    if self.regularization > 0:
                        gradient[1:] += (2 * self.regularization * self.coefficients[1:]) / len(batch_indices)
                    
                    self.coefficients -= self.alpha * gradient
            
            # Calculate and store cost
            y_pred = np.dot(X_normalized, self.coefficients)
            errors = y_pred - y
            
            if self.regularization > 0:
                reg_term = self.regularization * np.sum(self.coefficients[1:]**2)
                cost = np.mean(errors**2) + reg_term/n
            else:
                cost = np.mean(errors**2)
                
            self.cost_history.append(cost)
            
            # Early stopping
            if abs(prev_cost - cost) < self.tolerance:
                break
                
            prev_cost = cost
            
            # Check for divergence
            if np.isnan(cost) or np.isinf(cost):
                break

    def predict(self, X):
        X = np.array(X)
        X_normalized = self._normalize(X)
        X_normalized = self._add_intercept(X_normalized)
        
        return np.dot(X_normalized, self.coefficients)
    
    def mean_squared_error(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    def r_squared(self, y_true, y_pred):
        ss_total = np.sum((y_true - np.mean(y_true))**2)
        ss_residual = np.sum((y_true - y_pred)**2)
        return 1 - (ss_residual / ss_total)
    
    def plot_learning_curve(self, figsize=(10, 6)):
        plt.figure(figsize=figsize)
        plt.plot(range(len(self.cost_history)), self.cost_history)
        plt.xlabel('Epochs')
        plt.ylabel('Cost (MSE)')
        plt.title('Convergence of Gradient Descent')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def learning_rate_scheduler(self, initial_rate, decay_rate=0.95, decay_steps=100):
        self.alpha = initial_rate * (decay_rate ** (self.epochs / decay_steps))
        return self.alpha