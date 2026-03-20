import numpy as np
import torch as torch   
import math

from utils import ensure_numpy, ensure_torch


def get_powerlaw(P, exp, offset=3, normalize=True):
    pl = (offset+np.arange(P)) ** -exp
    if normalize:
        pl /= pl.sum()
    return pl


def get_hermite_polynomials():
    return {
        1: lambda x: x,
        2: lambda x: x**2 - 1,
        3: lambda x: x**3 - 3*x,
        4: lambda x: x**4 - 6*x**2 + 3,
        5: lambda x: x**5 - 10*x**3 + 15*x,
        6: lambda x: x**6 - 15*x**4 + 45*x**2 - 15,
        7: lambda x: x**7 - 21*x**5 + 105*x**3 - 105*x,
        8: lambda x: x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105,
        9: lambda x: x**9 - 36*x**7 + 378*x**5 - 1260*x**3 + 945*x,
        10: lambda x: x**10 - 45*x**8 + 630*x**6 - 3150*x**4 + 4725*x**2 - 945,
        11: lambda x: x**11 - 55*x**9 + 990*x**7 - 6930*x**5 + 17325*x**3 - 10395*x,
        12: lambda x: x**12 - 66*x**10 + 1485*x**8 - 13860*x**6 + 51975*x**4 - 62370*x**2 + 10395,
        13: lambda x: x**13 - 78*x**11 + 2145*x**9 - 25740*x**7 + 135135*x**5 - 270270*x**3 + 135135*x,
        14: lambda x: x**14 - 91*x**12 + 3003*x**10 - 45045*x**8 + 315315*x**6 - 945945*x**4 + 945945*x**2 - 135135,
        15: lambda x: x**15 - 105*x**13 + 4095*x**11 - 76545*x**9 + 675675*x**7 - 2702700*x**5 + 4729725*x**3 - 2027025*x,
        16: lambda x: x**16 - 120*x**14 + 5460*x**12 - 120120*x**10 + 1351350*x**8 - 7567560*x**6 + 18918900*x**4 - 20270250*x**2 + 34459425,
        17: lambda x: x**17 - 136*x**15 + 7140*x**13 - 180180*x**11 + 2297295*x**9 - 15315300*x**7 + 51081030*x**5 - 87513450*x**3 + 34459425*x,
        18: lambda x: x**18 - 153*x**16 + 9180*x**14 - 257400*x**12 + 3783780*x**10 - 30630600*x**8 + 122522400*x**6 - 229729500*x**4 + 172972500*x**2 - 34459425,
        19: lambda x: x**19 - 171*x**17 + 11628*x**15 - 375375*x**13 + 6432420*x**11 - 61261200*x**9 + 306306000*x**7 - 765765000*x**5 + 875134500*x**3 - 310134825*x,
        20: lambda x: x**20 - 190*x**18 + 14250*x**16 - 513513*x**14 + 10210200*x**12 - 117117000*x**10 + 765765000*x**8 - 2677114440*x**6 + 4670678100*x**4 - 3101348250*x**2 + 654729075,
    }


def compute_hermite_basis(X, monomials, is_X_PCAd=False):
    N, _ = X.shape
    if not is_X_PCAd:
        X = ensure_torch(X)
        U, _, _ = torch.linalg.svd(X, full_matrices=False)
        X = np.sqrt(N) * U
    X_PCA = ensure_numpy(X)

    hermites = get_hermite_polynomials()
    
    if type(monomials) != list:
        monomials = [monomials]

    H = np.zeros((N, len(monomials)))
    for i, monomial in enumerate(monomials):
        h = np.ones(N) / np.sqrt(N)
        for d_i, exp in monomial.items():
            d_i = int(d_i) #failsafe for when monomials aren't hard coded
            Z = np.sqrt(math.factorial(exp))
            h *= hermites[exp](X_PCA[:, d_i]) / Z
        H[:, i] = h
    return H


# for mlps

def get_synthetic_X(d=500, N=15000, offset=3, alpha=1.5, data_eigvals = None, gen=None, **kwargs):
    """
    Powerlaw synthetic data
    """
    data_eigvals = ensure_torch(get_powerlaw(d, alpha, offset=offset, normalize=True) if data_eigvals is None else data_eigvals)
    X = ensure_torch(torch.normal(0, 1, (N, d), generator=gen, device=data_eigvals.device)) * torch.sqrt(data_eigvals)
    return X, data_eigvals

def get_new_polynomial_data(lambdas, Vt, monomials, dim, N, data_eigvals, N_original,
                            new_X_fn=get_synthetic_X, coeffs=1, gen=None):
    new_X_args = dict(d=dim, N=N, data_eigvals=data_eigvals, gen=gen)
    X_new, _ = new_X_fn(**new_X_args)
    pca_x = X_new @ Vt.T @ torch.diag(lambdas**(-1.)) * N_original**(0.5)
    y_new = coeffs*compute_hermite_basis(X=pca_x, monomials=monomials, is_X_PCAd=True)*N**(0.5)
    if y_new.ndim == 2:
        y_new = y_new.sum(axis=1)/y_new.shape[1]
    return X_new, y_new


def polynomial_batch_fn(lambdas, Vt, monomials, bsz, data_eigvals, N,
                  X=None, y=None, data_creation_fn=get_new_polynomial_data, gen=None):
    lambdas, Vt, data_eigvals = map(ensure_torch, (lambdas, Vt, data_eigvals))
    dim = len(data_eigvals)
    def batch_fn(step: int, X=X, y=y):
        if (X is not None) and (y is not None):
            X_fixed = ensure_torch(X)
            y_fixed = ensure_torch(y)
            return X_fixed, y_fixed
        with torch.no_grad():
            dcf_args = dict(lambdas=lambdas, Vt=Vt, monomials=monomials, dim=dim,
                            N=bsz, data_eigvals=data_eigvals, N_original=N, gen=gen)
            X, y = data_creation_fn(**dcf_args)
        X, y = map(ensure_torch, (X, y))
        return X, y
    
    return batch_fn


# def get_synthetic_dataset(X=None, data_eigvals=None, d=500, N=15000, offset=3, alpha=1.5, cutoff_mode=10000,
#                           noise_size=0.1, yoffset=3, beta=1.2, normalized=True, gen=None, **kwargs):
#     """
#     noise_size: total noise size of the N-dim target vector y
#     """
#     if X is None:
#         X, data_eigvals = get_synthetic_X(d=d, N=N, offset=offset, alpha=alpha, gen=gen)

#     kernel_width = kwargs.get("kernel_width", 2)
#     kerneltype = kwargs.get("kerneltype", None)
#     hea_eigvals, monomials = generate_hea_monomials(data_eigvals, cutoff_mode, kerneltype.get_level_coeff_fn(kernel_width=kernel_width, data_eigvals=data_eigvals, **kwargs), kmax=kwargs.get('kmax', 9))
#     H = ensure_torch(compute_hermite_basis(X, monomials))
#     hea_eigvals = ensure_torch(hea_eigvals)
#     v_true = ensure_torch(get_powerlaw(H.shape[1], beta/2, offset=yoffset, normalize=normalized))
#     v_true = v_true if not normalized else v_true/torch.linalg.norm(v_true)* N**(0.5)
#     y = ensure_torch(H) @ v_true + ensure_torch(torch.normal(0., noise_size, (H.shape[0],), generator=gen, device=H.device))#/H.shape[0]**(0.5)
#     return X, y, H, monomials, hea_eigvals, v_true, data_eigvals


# leftover that might be useful later:
from scipy.special import zeta
def get_powerlaw_target(H, source_exp, offset=6, normalizeH=False, include_noise=False):
    if source_exp <= 1:
        raise ValueError("source_exp must be > 1 for powerlaw target")
    if offset < 1:
        raise ValueError("offset ≥ 1 required")
    M, P = H.shape
    if normalizeH:
        H = H / np.linalg.norm(H, axis=0, keepdims=True)
    squared_coeffs = get_powerlaw(P, source_exp, offset=offset)
    # Generate random signs for coefficients
    signs = -1 + 2*np.random.randint(0, 2, size=squared_coeffs.shape)
    coeffs = np.sqrt(squared_coeffs) * signs.astype(float)
    y = H @ coeffs
    if include_noise:
        totalsum = zeta(source_exp, offset)  # sum_{k=offset  }^infty k^{-exp}
        tailsum = zeta(source_exp, offset+P) # sum_{k=offset+P}^infty k^{-exp}
        noise_var = tailsum/(totalsum - tailsum)
        noise = np.random.normal(0, np.sqrt(noise_var / M), y.shape)
        # snr = y @ y / (noise @ noise)
        y /= np.linalg.norm(y)
        y += noise
    # we expect size(y_i) ~ 1
    y = np.sqrt(M) * y / np.linalg.norm(y)
    return y
