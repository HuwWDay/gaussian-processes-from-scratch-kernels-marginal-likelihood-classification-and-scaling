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

# Step 4 - linear_kernel
import numpy as np
from scipy.linalg import cho_solve, solve_triangular
from scipy.optimize import minimize


def as_2d(X):
    X = np.asarray(X, dtype=float)
    return X[:, None] if X.ndim == 1 else X


def linear_kernel(X1, X2, c=0.0, tau=1.0):
    # TODO: return tau * (X1 - c) @ (X2 - c).T for 2-D inputs.
    X1_2d, X2_2d = as_2d(X1), as_2d(X2)
    return tau* (X1_2d-c) @ (X2_2d - c).T

# Step 5 - kernel_from_spec
import numpy as np
from scipy.linalg import cho_solve, solve_triangular
from scipy.optimize import minimize


KERNELS = {"se": se_kernel, "matern": matern_kernel, "periodic": periodic_kernel, "linear": linear_kernel}


def kernel_from_spec(spec, X1, X2):
    """
    Evaluates a kernel built from a nested specification using the sum and product rules.
    """
    op = spec[0]
    
    if op == "+":
        # Sum of all child specs: sum their resulting (n1, n2) matrices
        return sum(kernel_from_spec(child, X1, X2) for child in spec[1:])
    
    elif op == "*":
        # Product of all child specs: multiply their resulting (n1, n2) matrices elementwise
        result = 1.0
        for child in spec[1:]:
            result = result * kernel_from_spec(child, X1, X2)
        return result
    
    else:
        # Leaf node: (name, params) or just (name,)
        name = op
        params = spec[1] if len(spec) > 1 else {}
        return KERNELS[name](X1, X2, **params)

# Step 6 - is_valid_gram
def is_valid_gram(K, tol=1e-8):
    """
    Checks if a matrix K is a valid Gram matrix (symmetric and positive semi-definite).
    Returns: (symmetric, min_eig, valid) as (bool, float, bool).
    """
    K = np.asarray(K)
    
    # Check symmetry: max absolute difference between K and its transpose <= 1e-10
    symmetric = bool(np.max(np.abs(K - K.T)) <= 1e-10)
    
    # Compute smallest eigenvalue of the symmetrized matrix
    K_sym = (K + K.T) / 2.0
    min_eig = float(np.min(np.linalg.eigvalsh(K_sym)))
    
    # Valid if it is symmetric and the minimum eigenvalue is >= -tol
    valid = bool(symmetric and min_eig >= -tol)
    
    return symmetric, min_eig, valid

# Step 7 - sample_gp_prior
def sample_gp_prior(K, n_samples, rng, jitter=1e-8):
    """
    Draws samples from a Gaussian Process prior with covariance matrix K.
    """
    n = K.shape[0]
    # Add jitter to the diagonal for numerical stability during Cholesky decomposition
    L = np.linalg.cholesky(K + jitter * np.eye(n))
    
    # Draw standard normal noise of shape (n, n_samples)
    Z = rng.standard_normal((n, n_samples))
    
    # Transform noise and transpose to get (n_samples, n)
    return (L @ Z).T

# Step 8 - gp_condition
import numpy as np
from scipy.linalg import cho_solve, solve_triangular
from scipy.optimize import minimize

def gp_condition(X, f, Xs, kernel, jitter=1e-10):
    """
    Computes the noiseless GP posterior mean and covariance conditioned on observations (X, f).
    
    mu  = Ks.T @ K^{-1} @ f
    cov = Kss - Ks.T @ K^{-1} @ Ks
    """
    X, Xs = np.asarray(X), np.asarray(Xs)
    N = X.shape[0]

    # Compute covariance matrices
    K = kernel(X, X) + jitter * np.eye(N)
    Ks = kernel(X, Xs)
    Kss = kernel(Xs, Xs)

    # Cholesky decomposition: K = L @ L.T
    L = np.linalg.cholesky(K)

    # Compute v = L^{-1} @ Ks so that v.T @ v = Ks.T @ K^{-1} @ Ks
    v = solve_triangular(L, Ks, lower=True)

    # Compute alpha = K^{-1} @ f = L.T^{-1} @ (L^{-1} @ f)
    alpha = solve_triangular(L.T, solve_triangular(L, f, lower=True), lower=False)

    # Posterior mean and covariance
    mu = Ks.T @ alpha
    cov = Kss - v.T @ v

    return mu, cov

