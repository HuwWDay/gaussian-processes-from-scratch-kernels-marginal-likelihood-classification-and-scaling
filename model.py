"""
Gaussian Processes from Scratch: Kernels, Marginal Likelihood, Classification and Scaling

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - se_kernel
import numpy as np
from scipy.linalg import cho_solve, solve_triangular
from scipy.optimize import minimize


def as_2d(X):
    X = np.asarray(X, dtype=float)
    return X[:, None] if X.ndim == 1 else X

def se_kernel(X1, X2, length_scale=1.0, signal_var=1.0):
    """
    Computes the squared exponential (RBF) kernel with ARD length scales.
    """
    # Scale inputs by length_scale (supports scalar or 1D array of shape (D,))
    X1_2d = as_2d(X1) / length_scale
    X2_2d = as_2d(X2) / length_scale
    
    # Compute pairwise squared Euclidean distances: ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a b^T
    d2 = (np.sum(X1_2d**2, axis=1, keepdims=True) + 
          np.sum(X2_2d**2, axis=1, keepdims=True).T - 
          2 * (X1_2d @ X2_2d.T))
    
    # Clip small negative values due to floating-point inaccuracies
    d2 = np.clip(d2, 0, None)
    
    # Return (n1, n2) kernel matrix
    return signal_var * np.exp(-0.5 * d2)

