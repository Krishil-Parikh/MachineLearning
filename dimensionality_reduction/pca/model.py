"""
Principal Component Analysis (PCA) from scratch.
"""

import numpy as np
from ...core.base_model import BaseModel

class PCA(BaseModel):
    """Principal Component Analysis."""
    
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None
    
    def fit(self, X, y=None):
        """Fit PCA to data."""
        # Center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # Compute covariance matrix
        cov_matrix = np.cov(X_centered.T)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # Sort by eigenvalues
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Select top n_components
        self.components = eigenvectors[:, :self.n_components].real
        self.explained_variance = eigenvalues[:self.n_components].real / np.sum(eigenvalues)
        
        return self
    
    def transform(self, X):
        """Project data onto principal components."""
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)
    
    def fit_transform(self, X, y=None):
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def predict(self, X):
        """Alias for transform."""
        return self.transform(X)
    
    def score(self, X, y):
        """Placeholder score."""
        return 0
