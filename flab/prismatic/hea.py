import heapq
import os
import pickle
import numpy as np
from devices import as_ndarray


HEA_COEFFS_DIR = os.path.join(os.path.dirname(__file__), 'hea-coeffs')
HEA_DATASETS = ['mnist', 'cifar5m', 'svhn']
# The six cross-kernel trials, in the order they index the stored coeff tensor's
# first (trial) axis: gaussian1, gaussian4, gaussian16, laplace1, laplace4, laplace16.
HEA_KERNELS = [f'{k}{w}' for k in ('gaussian', 'laplace') for w in (1, 4, 16)]


def get_coeffs(dataset, normalization=False, snr=2.0, kernel=None):
    """Load precomputed Hermite-basis (HEA) decomposition of a vision target function.

    Each dataset's 10 class-indicator targets were decomposed in a dataset-adapted
    multivariate Hermite polynomial basis (see the hea-data-coeffs experiment). A
    mode is a Hermite polynomial identified by its leading monomial {pca_index:
    exponent}; its coefficient for class ``c`` is the projection of the (centered,
    unit-norm) class-``c`` indicator target onto that basis function.

    The decomposition was repeated as six independent *trials* per (dataset,
    normalization): two kernels (gaussian, laplace) x three widths (1, 4, 16). The
    kernel/width only selects which monomials enter the basis, so a mode measured in
    several trials gives several estimates of the same coefficient. The stored bundle
    holds, over the ``P_TOT`` monomials selected in >= 2 trials (degree-sorted):

    - ``coeffs``  : masked array (n_trials=6, P_TOT, n_classes=10); the per-trial
                    coefficients, masked wherever a trial did not select that mode.
    - ``mean_power``: (P_TOT, n_classes); per class, the mean of the squared
                    coefficient over the trials that measured the mode.
    - ``cv_power`` : (P_TOT,); the coefficient of variation of the mode's total power
                    ``v_i^T v_i`` (summed over classes) across the measuring trials.
                    A small value means the trials agree, i.e. the mode is measured
                    with high signal-to-noise; a large value means it is noise-limited.

    This function returns one of two views of that bundle:

    - **Single trial** (``kernel`` given): the raw coefficients of one kernel/width
      trial. Use this to reproduce a specific decomposition.
    - **Cross-trial consensus** (``kernel=None``, the default): the modes whose power
      is reliably measured (``cv_power <= 1/snr``), together with their trial-averaged
      power. Use this when you want only the trustworthy structure and want to discard
      noise-limited modes; ``snr`` sets how strict "reliable" is.

    Args:
        dataset (str): one of 'mnist', 'cifar5m', 'svhn'.
        normalization (bool): if True, load the condition where image vectors were
            normalized to unit norm before decomposition; if False (default), the
            unnormalized condition.
        snr (float): signal-to-noise threshold for the consensus view. Modes with
            cross-trial CV of power <= 1/snr are kept (higher snr => stricter =>
            fewer, cleaner modes). Ignored when ``kernel`` is given. Default 2.0.
        kernel (str or None): if given (case-insensitive), select a single trial;
            one of 'gaussian1', 'gaussian4', 'gaussian16', 'laplace1', 'laplace4',
            'laplace16' (kernel name + width). If None (default), return the
            cross-trial consensus view instead.

    Returns:
        If ``kernel`` is given, a tuple ``(monomials, coeffs)``:
            monomials (list of dict): the ``P_MODES`` monomials this trial selected
                (those also present in >= 1 other trial), degree-sorted.
            coeffs (ndarray): float64 (P_MODES, 10); coeffs[i, c] is this trial's
                coefficient of monomial i for class c.
        If ``kernel`` is None, a tuple ``(monomials, mean_power)``:
            monomials (list of dict): the ``P_filt`` reliably-measured monomials
                (cross-trial CV <= 1/snr), degree-sorted.
            mean_power (ndarray): float64 (P_filt, 10); mean_power[i, c] is the
                trial-averaged squared coefficient of monomial i for class c.

    Wrap any returned monomial in a ``Monomial`` (from flab.prismatic.hea) to query
    degree or render it in latex.

    Raises:
        ValueError: if ``dataset`` is not supported, if ``kernel`` (lowercased) is
            not one of the six valid strings, or if ``snr`` is not positive.

    Examples:
        >>> # cross-trial consensus modes at the default snr=2 (CV <= 0.5)
        >>> monomials, mean_power = get_coeffs('mnist')
        >>> mean_power.shape[1]
        10
        >>> # stricter snr keeps fewer, cleaner modes
        >>> mono_hi, mp_hi = get_coeffs('mnist', snr=10.0)   # CV <= 0.1
        >>> len(mono_hi) <= len(monomials)
        True
        >>> # a single kernel trial's raw coefficients (snr is ignored here)
        >>> monomials, coeffs = get_coeffs('svhn', normalization=True, kernel='laplace16')
        >>> coeffs.shape == (len(monomials), 10)
        True
        >>> from flab.prismatic.hea import Monomial
        >>> Monomial(monomials[1]).degree()          # first non-constant mode
        1
    """
    if dataset not in HEA_DATASETS:
        raise ValueError(f"dataset '{dataset}' not supported. Choose from {HEA_DATASETS}")
    if snr <= 0:
        raise ValueError(f"snr must be positive, got {snr}")

    kind = 'normed' if normalization else 'raw'
    fn = os.path.join(HEA_COEFFS_DIR, f'{dataset}-{kind}.pkl')
    with open(fn, 'rb') as f:
        data = pickle.load(f)

    monomials = list(data['monomials'])
    coeffs = data['coeffs']            # masked (n_trials, P_TOT, n_classes)
    mean_power = np.asarray(data['mean_power'])  # (P_TOT, n_classes)
    cv_power = np.asarray(data['cv_power'])       # (P_TOT,)

    if kernel is not None:
        key = kernel.lower()
        if key not in HEA_KERNELS:
            raise ValueError(f"kernel '{kernel}' invalid. Choose from {HEA_KERNELS}")
        k = HEA_KERNELS.index(key)
        row = np.ma.getmaskarray(coeffs)[k, :, 0]  # True where trial k lacks the mode
        keep = ~row
        monomials = [m for m, kp in zip(monomials, keep) if kp]
        coeffs_k = np.asarray(coeffs.data[k])[keep].astype(np.float64)  # (P_MODES, 10)
        return monomials, coeffs_k

    keep = cv_power <= 1.0 / snr        # nan CVs (e.g. power-free modes) drop out
    monomials = [m for m, kp in zip(monomials, keep) if kp]
    mean_power_filtered = mean_power[keep].astype(np.float64)  # (P_filt, 10)
    return monomials, mean_power_filtered


class Monomial(dict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __lt__(self, other):
        assert isinstance(other, Monomial)
        return self.degree() < other.degree()

    def degree(self):
        if len(self) == 0:
            return 0
        return sum(self.values())

    def max_degree(self):
        if len(self) == 0:
            return 0
        return max(self.values())

    def copy(self):
        return Monomial(super().copy())

    def __str__(self) -> str:
        if self.degree() == 0:
            return "1"
        monostr = ""
        for idx, exp in self.items():
            expstr = f"^{exp}" if exp > 1 else ""
            monostr += f"x_{{{idx}}}{expstr}"
        return f"${monostr}$"

    def __repr__(self):
        return self.__str__()

    @classmethod
    def from_repr(cls, s: str) -> "Monomial":
        """
        Parse strings like '$x_{0}^2x_{3}x_{10}^5$' or '$1$' into a Monomial.
        No regex used. Strict about format produced by __repr__/__str__.
        """
        if not isinstance(s, str):
            raise TypeError("from_repr expects a string")

        s = s.strip()
        if s.startswith("$") and s.endswith("$"):
            s = s[1:-1]
        s = s.replace(" ", "")

        if s in {"", "1"}:
            return cls()

        i, n = 0, len(s)
        out = {}

        def expect(ch: str):
            nonlocal i
            if i >= n or s[i] != ch:
                raise ValueError(f"Expected '{ch}' at pos {i} in {s!r}")
            i += 1

        def read_digits() -> int:
            nonlocal i
            start = i
            while i < n and s[i].isdigit():
                i += 1
            if start == i:
                raise ValueError(f"Expected digits at pos {start} in {s!r}")
            return int(s[start:i])

        while i < n:
            # x_{idx}
            expect('x')
            expect('_')
            expect('{')
            idx = read_digits()
            expect('}')

            # optional ^exp
            exp = 1
            if i < n and s[i] == '^':
                i += 1
                exp = read_digits()

            out[idx] = out.get(idx, 0) + exp

        return cls(out)
    
    def basis_factors(self, include_one: bool = False, canonical: bool = True):
        """
        Return a list of unit-degree Monomials whose product equals this monomial.
        Example: Monomial({0: 2, 3: 1}) -> [Monomial({0:1}), Monomial({0:1}), Monomial({3:1})]
        If degree == 0, returns [] unless include_one=True (then [Monomial({})]).
        If canonical=True, factors are ordered by increasing variable index.
        """
        if self.degree() == 0:
            return [Monomial({})] if include_one else []

        items = sorted(self.items()) if canonical else self.items()
        factors = []
        for idx, exp in items:
            for _ in range(int(exp)):
                factors.append(Monomial({idx: 1}))
        return factors
    
    def basis(self, canonical: bool = True) -> dict:
        if self.degree() == 0:
            return {}

        items = sorted(self.items()) if canonical else self.items()
        return {idx: int(exp) for idx, exp in items}


def compute_hea_eigval(data_eigvals, monomial, eval_level_coeff):
    hea_eigval = eval_level_coeff(monomial.degree())
    for i, exp in monomial.items():
        hea_eigval *= data_eigvals[i] ** exp
    return hea_eigval


def generate_hea_monomials(data_eigvals, num_monomials, eval_level_coeff, kmax=10):
    """
    Generates HEA eigenvalues and monomials in canonical learning order.

    Args:
        data_eigvals (iterable): data covariance eigenvalues
        num_monomials (int): Number of monomials to generate.
        eval_level_coeff (function): Function to evaluate kernel level coefficients.
        kmax (int): Search monomials up to degree kmax

    Returns:
        - hea_eigvals (np.ndarray): Array of HEA eigenvalues.
        - monomials (list): List of generated monomials.
    """
    try:
        num_monomials = abs(int(num_monomials))
    except Exception as e:
        raise ValueError(f"type(num_monomials) must be int, not {type(num_monomials)}") from e
    assert num_monomials >= 1
    data_eigvals = as_ndarray(data_eigvals)
    d = len(data_eigvals)

    # populate priority queue with top monomial at each degree up to kmax
    pq = []
    pq_members = set()
    first_hea_eigval = compute_hea_eigval(data_eigvals, Monomial({}), eval_level_coeff)
    for k in range(1, kmax+1):
        monomial = Monomial({0: k})
        hea_eigval = compute_hea_eigval(data_eigvals, monomial, eval_level_coeff)
        # Each entry in the priority queue is (-hea_eigval, Monomial({idx:exp, ...}))
        pq.append((-hea_eigval, monomial))
        pq_members.add(repr(monomial))
    heapq.heapify(pq)
    
    monomials = [Monomial({})]
    hea_eigvals = [first_hea_eigval]
    for _ in range(num_monomials-1):
        if not pq:
            print("Warning: priority queue exhausted before reaching num_monomials.")
            return np.array(hea_eigvals), monomials
        neg_hea_eigval, monomial = heapq.heappop(pq)
        pq_members.remove(repr(monomial))
        hea_eigvals.append(-neg_hea_eigval)
        monomials.append(monomial)
        
        # generate successor monomials of same degree
        for idx in list(monomial.keys()):
            if idx + 1 < d:
                next_monomial = monomial.copy()
                next_monomial[idx] -= 1
                if next_monomial[idx] == 0:
                    del next_monomial[idx]
                next_monomial[idx + 1] = next_monomial.get(idx + 1, 0) + 1
                if repr(next_monomial) not in pq_members:
                    hea_eigval = compute_hea_eigval(data_eigvals, next_monomial, eval_level_coeff)
                    heapq.heappush(pq, (-hea_eigval, next_monomial))
                    pq_members.add(repr(next_monomial))

    return np.array(hea_eigvals), monomials