"""Search algorithms receive candidate coordinates and a callable, never a reference table."""

from dataclasses import dataclass
from time import perf_counter
import warnings

import numpy as np
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from threadpoolctl import threadpool_limits


def periodic_features(angles):
    radians = np.deg2rad(np.asarray(angles))
    return np.column_stack((np.sin(radians[:, 0]), np.cos(radians[:, 0]),
                            np.sin(radians[:, 1]), np.cos(radians[:, 1])))


def expected_improvement(mean, std, best):
    """Analytic expected improvement for minimization, with zero-variance limit."""
    mean, std = np.asarray(mean), np.asarray(std)
    improvement = best - mean
    result = np.maximum(improvement, 0.0)
    mask = std > 1e-12
    z = improvement[mask] / std[mask]
    result[mask] = improvement[mask] * norm.cdf(z) + std[mask] * norm.pdf(z)
    return result


@dataclass
class SearchResult:
    indices: np.ndarray
    energies: np.ndarray
    elapsed_seconds: np.ndarray
    fit_bound_warnings: int = 0


def run_search(oracle, candidates, initial_indices, budget, method, seed):
    """Count initial queries in budget and evaluate each chosen point on the oracle.

    Each method independently evaluates the shared initial indices. Runtime
    includes initial queries, oracle calls, GP fitting, and candidate selection;
    excludes embedding, scoring, plotting, and writing output.
    """
    candidates = np.asarray(candidates)
    initial_indices = list(map(int, initial_indices))
    if method not in {"random", "bayesian"}:
        raise ValueError("Unknown search method")
    if not initial_indices or len(set(initial_indices)) != len(initial_indices):
        raise ValueError("Initial indices must be nonempty and distinct")
    if not len(initial_indices) <= budget <= len(candidates):
        raise ValueError("Budget must include initialization and fit candidate count")
    if min(initial_indices) < 0 or max(initial_indices) >= len(candidates):
        raise ValueError("Initial index outside candidate grid")
    rng = np.random.default_rng(seed)
    features = periodic_features(candidates)
    indices, energies, elapsed = [], [], []
    unseen = np.ones(len(candidates), dtype=bool)
    bound_warnings = 0
    # Small covariance matrices run faster and more predictably with one BLAS thread.
    with threadpool_limits(limits=1):
        start = perf_counter()
        for step in range(budget):
            if step < len(initial_indices):
                idx = initial_indices[step]
            else:
                remaining = np.flatnonzero(unseen)
                if method == "random":
                    idx = int(rng.choice(remaining))
                else:
                    # One shared length scale on the two unit circles. Optimized
                    # afresh from observations only; bounds are fixed in advance.
                    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
                        length_scale=1.0, length_scale_bounds=(0.05, 5.0), nu=2.5
                    )
                    gp = GaussianProcessRegressor(
                        kernel=kernel, alpha=1e-8, normalize_y=True,
                        n_restarts_optimizer=2,
                        random_state=int(rng.integers(0, 2**31 - 1)),
                    )
                    with warnings.catch_warnings(record=True) as recorded:
                        warnings.simplefilter("always", ConvergenceWarning)
                        gp.fit(features[indices], energies)
                    bound_warnings += sum(
                        issubclass(w.category, ConvergenceWarning) for w in recorded
                    )
                    for warning in recorded:
                        if not issubclass(warning.category, ConvergenceWarning):
                            warnings.warn(str(warning.message), warning.category)
                    mean, std = gp.predict(features[remaining], return_std=True)
                    ei = expected_improvement(mean, std, min(energies))
                    # Deterministic grid-order tie breaking; tested points excluded.
                    idx = int(remaining[np.argmax(ei)])
            indices.append(idx)
            energies.append(float(oracle(candidates[idx])))
            unseen[idx] = False
            elapsed.append(perf_counter() - start)
    return SearchResult(np.array(indices), np.array(energies),
                        np.array(elapsed), bound_warnings)
