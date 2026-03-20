import numpy as np
from scipy.optimize import bisect
import torch
from devices import as_ndarray, as_tensor


def krr(K, y, n_train, n_test, ridge=0):
    """Compute kernel ridge regression using the first n_train idxs of K and y
    as training samples and the last n_test idxs as test samples.
    
    Note: for ensemble statistics over multiple trials, use krr_resample.

    Args:
        K (tensor): Kernel matrix of shape (n_tot, n_tot).
        y (tensor): Labels of shape (n_tot,).
        n_train (int): Training set = first n_train idxs of K and y.
        n_test (int): Test set = last n_test idxs of K and y.
        ridge (float): regularized kernel = K_train + ridge * I_ntrain.

    Returns:
        Tuple of ndarrays ((y_hat_test, y_test), (y_hat_train, y_train)) where:
        (y_hat_test, y_test): Predictions and ground-truth for the test set.
        (y_hat_train, y_train): Predictions and ground-truth for the training set.
    """
    K = as_tensor(K)
    y = as_tensor(y)
    n_tot = K.shape[0]
    assert n_train + n_test <= n_tot

    K_train = K[:n_train, :n_train]
    y_train = y[:n_train]
    K_test = K[n_tot - n_test:, :n_train]
    y_test = y[n_tot - n_test:]

    if ridge == 0:
        alpha = torch.linalg.lstsq(K_train, y_train)
    else:
        eye = as_tensor(torch.eye(n_train))
        alpha = torch.linalg.lstsq(K_train + ridge * eye, y_train)

    y_hat_train = K_train @ alpha.solution
    y_hat_test = K_test @ alpha.solution

    return (y_hat_test, y_test), (y_hat_train, y_train)


def krr_resample(K, y, n_train, n_test, ntrials, ridge=0):
    """Run multiple trials of kernel ridge regression and return MSEs.

    Resamples data across trials by iterating sequentially through the
    idxs of K and y using a sliding window of size n_train + n_test.
    If multiple sweeps are needed, the idxs are shuffled between sweeps.

    Args:
        K (tensor): Kernel matrix of shape (n_tot, n_tot).
        y (tensor): Labels of shape (n_tot,).
        n_train (int): Number of training samples per trial.
        n_test (int): Number of test samples per trial.
        ntrials (int): Number of trials to run.
        ridge (float): regularized kernel = K_train + ridge * I_ntrain.

    Returns:
        test_mses (ndarray): Array of shape (ntrials,) with per-trial test MSE.
        train_mses (ndarray): Array of shape (ntrials,) with per-trial train MSE.
    """

    def shuffle_indices(n_tot, K, y):
        slc = torch.randperm(n_tot)
        K = K[slc[:, None], slc[None, :]]
        y = y[slc]
        return K, y
    
    K = as_tensor(K)
    y = as_tensor(y)
    n_tot = K.shape[0]
    n_per_trial = n_train + n_test
    assert n_per_trial <= n_tot
    
    cur_idx = 0
    train_mses = []
    test_mses = []
    for _ in range(ntrials):
        if cur_idx + n_per_trial > n_tot:
            K, y = shuffle_indices(n_tot, K, y)
            cur_idx = 0
        end_idx_train = cur_idx + n_train
        end_idx_test = cur_idx + n_per_trial
        
        K_train = K[cur_idx:end_idx_train, cur_idx:end_idx_train]
        K_test = K[cur_idx:end_idx_test, cur_idx:end_idx_train]
        y_train = y[cur_idx:end_idx_train]
        y_test = y[end_idx_train:end_idx_test]

        if ridge == 0:
            alpha = torch.linalg.lstsq(K_train, y_train)
        else:
            eye = as_tensor(torch.eye(n_train))
            alpha = torch.linalg.lstsq(K_train + ridge * eye, y_train)

        y_hat = K_test @ alpha.solution
        y_hat_train = y_hat[:n_train]
        y_hat_test = y_hat[n_train:]

        train_mse = ((y_hat_train - y_train)**2).mean(axis=-1).item()
        test_mse = ((y_hat_test - y_test)**2).mean(axis=-1).item()

        train_mses.append(train_mse)
        test_mses.append(test_mse)

        cur_idx += n_per_trial

    train_mses = np.array(train_mses)
    test_mses = np.array(test_mses)
    return test_mses, train_mses


def estimate_kappa(K, n, ridge=0):
    """Estimate kappa empirically from a random n x n submatrix of K
    using kappa_hat = 1 / Tr((K_n + ridge*I)^{-1}).
    
    Args:
        K (tensor): Kernel matrix of shape (n_tot, n_tot).
        n (int): Size of the random submatrix to use.
        ridge (float): regularized kernel = K_n + ridge * I_n.

    Returns:
        kappa_hat (float): estimate of kappa.
    """
    K = as_tensor(K)
    n_tot = K.shape[0]
    slc = torch.randperm(n_tot)[:n]
    K_n = K[slc[:, None], slc[None, :]]

    if ridge != 0:
        eye = as_tensor(torch.eye(n))
        K_n = K_n + ridge * eye
    kappa_hat = 1 / torch.trace(torch.linalg.inv(K_n)).item()
    return kappa_hat


def compute_learnabilities(n, eigvals, ridge=0):
    """Compute per-mode learnabilities from the eigenlearning conservation law.
    Vectorized implementation that can evaluate multiple n values.

    Implicitly solves for kappa satisfying: n = sum(eigvals / (eigvals + kappa)) + ridge/kappa.
    Returns learnabilities = eigvals / (eigvals + kappa).

    Args:
        n (int or ndarray): Number of training samples. If array, compute learnabilities for each n.
        eigvals (ndarray): Kernel eigenvalues, shape (n_modes,). Must satisfy n < len(eigvals).
        ridge (float): regularized kernel = K_train + ridge * I_n.

    Returns:
        learnabilities (ndarray): Array of shape (n_modes,) if n is scalar, or
            (len(n), n_modes) if n is a 1D array.
        kappa (float or ndarray): Solved kappa values (for each n).
    """

    def _compute_lrn(n, eigvals, ridge):
        conservation_law = lambda kap: (eigvals/(eigvals+kap)).sum() + ridge/kap - n
        kappa = bisect(conservation_law, 1e-25, 1e10, maxiter=128)
        learnabilities = eigvals / (eigvals + kappa)
        return learnabilities, kappa
    
    n = as_ndarray(n)
    eigvals = as_ndarray(eigvals)
    if len(eigvals) <= np.max(n):
        raise ValueError("n_train must be less than the number of modes")
    if np.ndim(n) == 0:
        return _compute_lrn(float(n), eigvals, ridge)
    if np.ndim(n) == 1:
        learnabilities = np.zeros((len(n), len(eigvals)))
        kappas = np.zeros(len(n))
        for i in range(len(n)):
            learnabilities[i], kappas[i] = _compute_lrn(n[i], eigvals, ridge)
        return learnabilities, kappas
    else:
        raise ValueError("n must be a scalar or 1D array")


def compute_learning_curve(n, learnabilities, eigcoeffs, noise_var=0):
    """Compute the theoretical learning curve (test MSE as a function of n_train).
    Vectorized implementation that can evaluate multiple n values.    

    Uses the eigenlearning framework:
        e0 = n / (n - sum(learnabilities^2))  # overfitting coefficient
        test_mse = e0 * (sum((1 - learnabilities)^2 * eigcoeffs^2) + noise_var)
    Learnabilities are truncated if not enough eigcoeffs are provided.

    Args:
        n (int or ndarray): Number of training samples. Scalar or 1D array of shape (n_sizes,).
        learnabilities (ndarray): Per-mode learnabilities, shape (n_modes,) or (n_sizes, n_modes).
        eigcoeffs (ndarray): Target function coefficients in the eigenbasis, shape (n_modes,).
        noise_var (float): Label noise variance.

    Returns:
        test_mse (float or ndarray): Scalar or 1D array of predicted test MSE values.
    """
    n = as_ndarray(n)
    if len(eigcoeffs) < learnabilities.shape[-1]:
        learnabilities = learnabilities[..., :len(eigcoeffs)]
    sum_axis = None if learnabilities.ndim == 1 else 1
    e0 = n / (n - (learnabilities**2).sum(axis=sum_axis))
    test_mse = e0 * (((1-learnabilities)**2 * eigcoeffs**2).sum(axis=sum_axis) + noise_var)
    return test_mse


def eigenlearning(n, eigvals, eigcoeffs, ridge=0, noise_var=0):
    """Compute the full eigenlearning predictions for KRR.
    Vectorized implementation that can evaluate multiple n values.

    Args:
        n (int or ndarray): Number of training samples. Scalar or 1D array.
        eigvals (ndarray): Kernel eigenvalues, shape (n_modes,). Must satisfy n < len(eigvals).
        eigcoeffs (ndarray): Target function coefficients in the eigenbasis, shape (n_modes,).
        ridge (float): regularized kernel = K_train + ridge * I_n.
        noise_var (float): Label noise variance.

    Returns:
        dict with keys:
            kappa (float or ndarray): Solved kappa value(s).
            learnabilities (ndarray): Per-mode learnabilities.
            train_mse (float or ndarray): Predicted training MSE.
            test_mse (float or ndarray): Predicted test MSE.
    """
    n = as_ndarray(n)
    learnabilities, kappa = compute_learnabilities(n, eigvals, ridge)
    test_mse = compute_learning_curve(n, learnabilities, eigcoeffs, noise_var)
    train_mse = (ridge / (n * kappa))**2 * test_mse

    return {
        "kappa": kappa,
        "learnabilities": learnabilities,
        "train_mse": train_mse,
        "test_mse": test_mse,
    }
