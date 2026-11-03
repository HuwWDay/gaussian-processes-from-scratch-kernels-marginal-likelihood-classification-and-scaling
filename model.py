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

# Step 2 - matern_kernel
import numpy as np
from scipy.linalg import cho_solve, solve_triangular
from scipy.optimize import minimize


def as_2d(X):
    X = np.asarray(X, dtype=float)
    return X[:, None] if X.ndim == 1 else X


def matern_kernel(X1, X2, nu=1.5, length_scale=1.0, signal_var=1.0):
    # TODO: r = pairwise Euclidean distance / length_scale (clip squared distances at 0 first);
    #       closed forms for nu = 0.5, 1.5, 2.5 (eq 17.11-17.13); ValueError for any other nu.
    X1_2d, X2_2d = as_2d(X1), as_2d(X2)
    r = (np.sum(X1_2d**2, axis=1, keepdims=True) + 
          np.sum(X2_2d**2, axis=1, keepdims=True).T - 
          2 * (X1_2d @ X2_2d.T))
    
    # Clip small negative values due to floating-point inaccuracies
    r = np.sqrt(np.clip(r, 0, None))

    if nu == 0.5:
        return signal_var*np.exp(-r/length_scale)
    elif nu == 1.5:
        return signal_var*(1+np.sqrt(3)*r/length_scale)*np.exp(-np.sqrt(3)*r/length_scale)
    elif nu == 2.5:
        return signal_var*(1+np.sqrt(5)*r/length_scale + 5*r**2 / (3*length_scale**2))*np.exp(-np.sqrt(5)*r / length_scale)
    else:
        raise ValueError

# Step 3 - periodic_kernel
import numpy as np
from scipy.linalg import cho_solve, solve_triangular
from scipy.optimize import minimize


def as_2d(X):
    X = np.asarray(X, dtype=float)
    return X[:, None] if X.ndim == 1 else X


def periodic_kernel(X1, X2, length_scale=1.0, period=1.0, signal_var=1.0):
    # TODO: r = pairwise Euclidean distance; return signal_var * exp(-2 sin^2(pi r / period) / length_scale^2).
    # Compute pairwise squared Euclidean distances: ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a b^T
    X1_2d, X2_2d = as_2d(X1), as_2d(X2)
    r = (np.sum(X1_2d**2, axis=1, keepdims=True) + 
          np.sum(X2_2d**2, axis=1, keepdims=True).T - 
          2 * (X1_2d @ X2_2d.T))
    
    # Clip small negative values due to floating-point inaccuracies
    r = np.sqrt(np.clip(r, 0, None))
    return signal_var*np.exp(-2/length_scale**2 * np.sin(np.pi*r / period)**2)

