import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

class SVM():
    def __init__(self, lr=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = lr
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        samples_x, features_x = X.shape
        self.w = np.zeros(features_x)
        self.b = 0

        for _ in range(self.n_iters):
            for index, x_i in enumerate(X):
                condition = y[index] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(y[index], x_i))
                    self.b -= self.lr * y[index]

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)

    def accuracy(self, X, y):
        predictions = self.predict(X)
        correct_predictions = np.sum(predictions == y)
        accuracy = correct_predictions / len(y)
        return accuracy

    def error(self, X, y):
        return 1 - self.accuracy(X, y)

    def plot_decision_boundary(self, X, y):
        plt.figure(figsize=(10, 6))

        plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='b', label='Class 1', alpha=0.5)
        plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], color='r', label='Class -1', alpha=0.5)

        x_1 = np.linspace(min(X[:, 0]) - 1, max(X[:, 0]) + 1, 100)
        x_2 = np.linspace(min(X[:, 1]) - 1, max(X[:, 1]) + 1, 100)
        X1, X2 = np.meshgrid(x_1, x_2)
        Z = np.dot(np.c_[X1.ravel(), X2.ravel()], self.w) - self.b
        Z = Z.reshape(X1.shape)

        plt.contour(X1, X2, Z, levels=[0], linewidths=2, colors='black')
        plt.contour(X1, X2, Z, levels=[-1, 1], linewidths=1, linestyles='dashed', colors='gray')

        plt.title("SVM Decision Boundary")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()
        plt.grid()
        plt.show()

if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv("Student_Performance.csv")

    df["Extracurricular Activities"] = LabelEncoder().fit_transform(df["Extracurricular Activities"])
    y = df["Extracurricular Activities"]
    df = df.drop(columns=["Extracurricular Activities"])
    X = df.iloc[:, :].values

    # Convert target variable to -1 and 1 (SVM requires this)
    y = np.where(y == 0, -1, 1)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Hyperparameters tuning
    lr = 0.01  # Increased learning rate
    lambda_param = 0.001  # Reduced regularization
    n_iters = 5000  # Increased iterations for better convergence

    # Initialize and train the SVM model
    model = SVM(lr=lr, lambda_param=lambda_param, n_iters=n_iters)
    model.fit(X_train, y_train)

    # Predictions and Accuracy
    accuracy = model.accuracy(X_test, y_test)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    # Plot Decision Boundary (for 2D data)
    if X_train.shape[1] == 2:  # Only if 2 features for visualization
        model.plot_decision_boundary(X_train, y_train)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tqdm import tqdm

class SVM:
    def __init__(self, lr=0.001, lambda_param=0.01, n_iters=1000, batch_size=None, tol=1e-5):
        self.lr = lr
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None
        self.batch_size = batch_size
        self.tol = tol
        self.losses = []

    def _compute_cost(self, X, y):
        n_samples = X.shape[0]
        
        # Calculate the hinge loss
        scores = y * (np.dot(X, self.w) - self.b)
        hinge_loss = np.maximum(0, 1 - scores)
        
        # L2 regularization
        reg_loss = self.lambda_param * np.sum(self.w ** 2)
        
        # Total cost
        cost = reg_loss + np.sum(hinge_loss) / n_samples
        return cost

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        
        # Determine batch size
        batch_size = n_samples if self.batch_size is None else min(self.batch_size, n_samples)
        
        # Previous cost for convergence check
        prev_cost = float('inf')
        
        # Training with early stopping
        for epoch in range(self.n_iters):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Mini-batch processing
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Vectorized computation for the batch
                scores = y_batch * (np.dot(X_batch, self.w) - self.b)
                mask = scores < 1
                
                # Gradient calculation
                dw = 2 * self.lambda_param * self.w
                if np.any(mask):
                    X_masked = X_batch[mask]
                    y_masked = y_batch[mask]
                    dw -= np.sum(np.outer(y_masked, np.ones(n_features)) * X_masked, axis=0) / batch_size
                    db = -np.sum(y_masked) / batch_size
                else:
                    db = 0
                
                # Weight update
                self.w -= self.lr * dw
                self.b -= self.lr * db
            
            # Compute cost and check for convergence
            cost = self._compute_cost(X, y)
            self.losses.append(cost)
            
            # Early stopping check
            if abs(prev_cost - cost) < self.tol:
                break
            prev_cost = cost
            
            # Learning rate decay
            self.lr *= 0.999

    def predict(self, X):
        decision_values = np.dot(X, self.w) - self.b
        return np.sign(decision_values)
    
    def decision_function(self, X):
        return np.dot(X, self.w) - self.b

    def accuracy(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)

    def plot_training_curve(self):
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.losses)), self.losses)
        plt.title('SVM Training Curve')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_decision_boundary(self, X, y):
        plt.figure(figsize=(10, 6))
        
        # Plot the data points
        plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='b', label='Class 1', alpha=0.5)
        plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], color='r', label='Class -1', alpha=0.5)
        
        # Create a mesh grid
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                            np.linspace(y_min, y_max, 100))
        
        # Get Z values
        Z = self.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary and margins
        plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')
        plt.contour(xx, yy, Z, levels=[-1, 1], linewidths=1, linestyles='dashed', colors='gray')
        
        # Fill the contours
        plt.contourf(xx, yy, Z, levels=[-float('inf'), 0, float('inf')],
                    colors=['#FFAAAA', '#AAAAFF'], alpha=0.3)
        
        plt.title("SVM Decision Boundary")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()