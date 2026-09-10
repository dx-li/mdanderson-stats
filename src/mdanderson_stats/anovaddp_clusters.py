"""Covariate-dependent Gaussian atom and cluster updates for ANOVA DDP."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import logsumexp

from ._cdflib import _freeze
from ._validation import FloatArray, count, finite, scalar


@dataclass(frozen=True)
class AnovaDDPAtomPosterior:
    mean: FloatArray
    covariance: FloatArray


@dataclass(frozen=True)
class AnovaDDPClusters:
    labels: NDArray[np.int64]
    atoms: FloatArray
    counts: NDArray[np.int64]
    created: int
    removed: int
    assignment_uniforms: FloatArray
    atom_normal_draws: FloatArray


def _spd(value: ArrayLike, dimension: int, name: str) -> tuple[FloatArray, FloatArray]:
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    c = finite(value, name)
    if c.shape != (dimension, dimension) or not np.allclose(c, c.T, rtol=1e-12, atol=0):
        raise ValueError(f"{name} must be symmetric with shape ({dimension},{dimension})")
    c = c / 2 + c.T / 2
    try:
        chol = np.linalg.cholesky(c)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} must be positive definite") from exc
    return c, chol


def _inputs(
    parameters: ArrayLike,
    design: ArrayLike,
    residual_covariance: ArrayLike,
    base_mean: ArrayLike,
    base_covariance: ArrayLike,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    if any(np.iscomplexobj(a) for a in (parameters, design, base_mean)):
        raise ValueError("conditional parameters, design and base mean must be real")
    y, x, m = (
        finite(a, n)
        for a, n in zip((parameters, design, base_mean), ("parameters", "design", "base_mean"))
    )
    if (
        y.ndim != 2
        or not y.shape[0]
        or not y.shape[1]
        or x.ndim != 2
        or x.shape[0] != y.shape[0]
        or not x.shape[1]
    ):
        raise ValueError(
            "require aligned nonempty subject-by-parameter and subject-by-covariate matrices"
        )
    dimension = x.shape[1] * y.shape[1]
    if dimension > 100 or y.shape[0] > 10000 or m.shape != (dimension,) or np.any(x[:, 0] != 1):
        raise ValueError(
            "require <=100 atom parameters, <=10000 subjects, "
            "matching base mean and intercept column one"
        )
    s, sl = _spd(residual_covariance, y.shape[1], "residual_covariance")
    c, _ = _spd(base_covariance, dimension, "base_covariance")
    return y, x, m, s, sl, c


def _atom(
    y: FloatArray, x: FloatArray, s: FloatArray, m: FloatArray, c: FloatArray
) -> AnovaDDPAtomPosterior:
    d = s.shape[0]
    prior_precision = np.linalg.solve(c, np.eye(m.size))
    precision = prior_precision.copy()
    rhs = prior_precision @ m
    for yi, xi in zip(y, x, strict=True):
        f = np.kron(xi[None, :], np.eye(d))
        solved = np.linalg.solve(s, f)
        precision += f.T @ solved
        rhs += solved.T @ yi
    covariance = np.linalg.solve(precision, np.eye(m.size))
    mean = np.linalg.solve(precision, rhs)
    covariance = covariance / 2 + covariance.T / 2
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(covariance)):
        raise ArithmeticError("Gaussian atom posterior exceeds floating-point range")
    return AnovaDDPAtomPosterior(_freeze(mean), _freeze(covariance))


def anovaddp_atom_posterior(
    conditional_parameters: ArrayLike,
    design: ArrayLike,
    *,
    residual_covariance: ArrayLike,
    base_mean: ArrayLike,
    base_covariance: ArrayLike,
) -> AnovaDDPAtomPosterior:
    """Gaussian coefficient posterior for all subjects assigned to one atom.

    Atom vector is covariate-major: q consecutive blocks of d parameters, with
    subject mean (design_i kron I_d) @ atom. Native dimensions are d=5,q=7.
    Input parameters are the conditional random effects, not raw six-vectors.
    """
    y, x, m, s, _, c = _inputs(
        conditional_parameters, design, residual_covariance, base_mean, base_covariance
    )
    return _atom(y, x, s, m, c)


def _logpdf(y: FloatArray, mean: FloatArray, chol: FloatArray) -> float:
    z = np.linalg.solve(chol, y - mean)
    value = float(-0.5 * (z @ z) - np.log(np.diag(chol)).sum() - 0.5 * y.size * np.log(2 * np.pi))
    if not np.isfinite(value):
        raise ArithmeticError("Gaussian cluster log density exceeds floating-point range")
    return value


def anovaddp_cluster_sweep(
    conditional_parameters: ArrayLike,
    design: ArrayLike,
    labels: ArrayLike,
    atoms: ArrayLike,
    *,
    residual_covariance: ArrayLike,
    base_mean: ArrayLike,
    base_covariance: ArrayLike,
    concentration: float,
    seed: int | None = None,
) -> AnovaDDPClusters:
    """Sequential assignment sweep followed by Gaussian draws for all atoms.

    Existing-cluster weight is its leave-one-out count times Gaussian density.
    New-cluster weight integrates the Gaussian base measure. Empty clusters and
    their atom rows are removed together, repairing native label/atom drift.
    Labels are contiguous zero-based integers; atoms have shape (clusters,q*d).
    This is the cluster block, not a complete DDP chain.
    """
    y, x, m, s, sl, c = _inputs(
        conditional_parameters, design, residual_covariance, base_mean, base_covariance
    )
    if np.iscomplexobj(labels) or np.iscomplexobj(atoms):
        raise ValueError("labels and atoms must be real")
    raw = count(labels, "labels")
    a = finite(atoms, "atoms").copy()
    if (
        raw.shape != (y.shape[0],)
        or a.ndim != 2
        or a.shape[1] != m.size
        or not a.shape[0]
        or np.any(raw >= a.shape[0])
    ):
        raise ValueError("labels must index the supplied nonempty atom matrix")
    lab = raw.astype(np.int64)
    sizes = np.bincount(lab, minlength=a.shape[0])
    if np.any(sizes == 0):
        raise ValueError("every initial atom must have an occupied cluster")
    mass = scalar(concentration, "concentration")
    if mass <= 0 or (
        seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0)
    ):
        raise ValueError("positive concentration and nonnegative integer seed required")
    rng = np.random.default_rng(seed)
    created = removed = 0
    uniforms = np.empty(y.shape[0])
    innovations: list[FloatArray] = []
    for i in range(y.shape[0]):
        previous = lab[i]
        sizes[previous] -= 1
        lab[i] = -1
        if sizes[previous] == 0:
            sizes = np.delete(sizes, previous)
            a = np.delete(a, previous, axis=0)
            lab[lab > previous] -= 1
            removed += 1
        f = np.kron(x[i : i + 1], np.eye(y.shape[1]))
        _, predictive_chol = _spd(s + f @ c @ f.T, y.shape[1], "predictive covariance")
        standardized = np.linalg.solve(sl, (y[i] - a @ f.T).T)
        existing = (
            np.log(sizes)
            - 0.5 * np.sum(standardized**2, axis=0)
            - np.log(np.diag(sl)).sum()
            - 0.5 * y.shape[1] * np.log(2 * np.pi)
        )
        if not np.all(np.isfinite(existing)):
            raise ArithmeticError("Gaussian cluster log density exceeds floating-point range")
        log_weights = np.r_[existing, np.log(mass) + _logpdf(y[i], f @ m, predictive_chol)]
        probabilities = np.exp(log_weights - logsumexp(log_weights))
        probabilities /= probabilities.sum()
        uniforms[i] = rng.random()
        cumulative = np.cumsum(probabilities)
        cumulative[-1] = 1.0
        selected = int(np.searchsorted(cumulative, uniforms[i], side="right"))
        if selected == sizes.size:
            posterior = _atom(y[i : i + 1], x[i : i + 1], s, m, c)
            normal = rng.standard_normal(m.size)
            innovations.append(normal)
            new = posterior.mean + np.linalg.cholesky(posterior.covariance) @ normal
            a = np.vstack((a, new))
            sizes = np.r_[sizes, 0]
            created += 1
        lab[i] = selected
        sizes[selected] += 1
    for j in range(sizes.size):
        mask = lab == j
        posterior = _atom(y[mask], x[mask], s, m, c)
        normal = rng.standard_normal(m.size)
        innovations.append(normal)
        a[j] = posterior.mean + np.linalg.cholesky(posterior.covariance) @ normal
    if not np.all(np.isfinite(a)):
        raise ArithmeticError("Gaussian atom draw exceeds floating-point range")
    return AnovaDDPClusters(
        np.frombuffer(lab.tobytes(), dtype=np.int64),
        _freeze(a),
        np.frombuffer(sizes.tobytes(), dtype=np.int64),
        created,
        removed,
        _freeze(uniforms),
        _freeze(np.stack(innovations)),
    )
