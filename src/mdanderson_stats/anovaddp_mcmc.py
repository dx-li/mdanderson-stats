"""MCMC orchestration for the supplied nonlinear ANOVA DDP model."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._cdflib import _freeze
from ._validation import FloatArray, count, finite, scalar
from .anovaddp import anovaddp_loglikelihood
from .anovaddp_clusters import _spd, anovaddp_cluster_sweep
from .anovaddp_hyperparameters import anovaddp_hyperparameter_update
from .anovaddp_updates import anovaddp_subject_update, anovaddp_variance_posterior


@dataclass(frozen=True)
class AnovaDDPFit:
    """Saved states after complete sweeps; varying-size atom matrices are a tuple.

    Label identities are arbitrary between draws. Acceptance fractions cover all
    iterations, including warmup, and do not establish MCMC convergence.
    """

    parameters: FloatArray
    observation_variance: FloatArray
    residual_covariance: FloatArray
    base_mean: FloatArray
    base_covariance: FloatArray
    concentration: FloatArray
    labels: NDArray[np.int64]
    atoms: tuple[FloatArray, ...]
    cluster_count: NDArray[np.int64]
    log_likelihood: FloatArray
    acceptance_fraction: FloatArray
    saved_iterations: NDArray[np.int64]
    variance_mode: str


def fit_anovaddp(
    time: ArrayLike,
    observations: ArrayLike,
    subject: ArrayLike,
    design: ArrayLike,
    *,
    initial_parameters: ArrayLike,
    initial_covariance: ArrayLike,
    base_prior: ArrayLike,
    base_covariance: ArrayLike,
    covariance_df: int = 12,
    alpha0: float = 4.25,
    beta0: float = 1.25,
    concentration_shape: float = 0.05,
    concentration_rate: float = 0.05,
    iterations: int = 2000,
    burn_in: int = 1000,
    thin: int = 1,
    variance_mode: Literal["documented", "source"] = "documented",
    seed: int | None = None,
) -> AnovaDDPFit:
    """Fit the six-parameter nonlinear ANOVA DDP model using all Gibbs blocks.

    Subject IDs are zero-based rows of design/initial_parameters. Start with one
    occupied atom equal to base_prior and concentration one, as in the archive.
    Defaults for hyperparameters/iterations follow the supplied demonstration.
    Corrects singleton bookkeeping, the discarded S update and (by default)
    the omitted variance-prior scale. The base intercept block is independent
    of other base-covariance blocks. Prediction/reporting are separate workflows.
    """
    for name, value, lower, upper in (
        ("iterations", iterations, 1, 100000),
        ("burn_in", burn_in, 0, iterations - 1),
        ("thin", thin, 1, iterations),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
            raise ValueError(f"{name} must be an integer in [{lower},{upper}]")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
        raise ValueError("seed must be a nonnegative integer or None")
    if any(
        np.iscomplexobj(a)
        for a in (time, observations, subject, design, initial_parameters, base_prior)
    ):
        raise ValueError("data, design and parameters must be real")
    t, y, x, theta, b = (
        finite(v, n)
        for v, n in zip(
            (time, observations, design, initial_parameters, base_prior),
            ("time", "observations", "design", "initial_parameters", "base_prior"),
        )
    )
    ids = count(subject, "subject")
    if (
        theta.ndim != 2
        or theta.shape[1] != 6
        or not 1 <= theta.shape[0] <= 10000
        or x.ndim != 2
        or x.shape[0] != theta.shape[0]
        or not 1 <= x.shape[1] <= 20
        or np.any(x[:, 0] != 1)
    ):
        raise ValueError("require (subjects,6) parameters and aligned design with intercept one")
    n = theta.shape[0]
    dimension = 5 * x.shape[1]
    if (
        t.ndim != 1
        or not t.size
        or y.shape != t.shape
        or ids.shape != t.shape
        or np.any(ids >= n)
        or b.shape != (dimension,)
    ):
        raise ValueError("require aligned observation vectors, valid subject IDs and q*5 base mean")
    integer_ids = ids.astype(np.int64)
    if np.any(np.bincount(integer_ids, minlength=n) == 0):
        raise ValueError("each design row must have observations")
    s0, _ = _spd(initial_covariance, 6, "initial_covariance")
    c, _ = _spd(base_covariance, dimension, "base_covariance")
    if np.any(c[:5, 5:] != 0):
        raise ValueError("base intercept block must be independent of other coefficient blocks")
    if (
        isinstance(covariance_df, bool)
        or not isinstance(covariance_df, int)
        or not 6 <= covariance_df <= 1000000
    ):
        raise ValueError("covariance_df must be an integer in [6,1000000]")
    a0, b0, am, bm = (
        scalar(v, nm)
        for v, nm in zip(
            (alpha0, beta0, concentration_shape, concentration_rate),
            ("alpha0", "beta0", "concentration_shape", "concentration_rate"),
        )
    )
    if min(a0, b0, am, bm) <= 0 or variance_mode not in ("documented", "source"):
        raise ValueError(
            "positive prior parameters and variance_mode documented or source required"
        )
    saved = np.arange(burn_in + 1, iterations + 1, thin, dtype=np.int64)
    size = saved.size
    bound = size * (6 * n + n + 36 + dimension + dimension**2 + n * dimension + 3)
    if bound > 50_000_000 or n * iterations > 5_000_000 or t.size * iterations > 100_000_000:
        raise ValueError(
            "chain exceeds 50M saved values, 5M subject sweeps or 100M observation sweeps"
        )
    order = np.argsort(integer_ids, kind="stable")
    ts, ys = t[order], y[order]
    starts = np.r_[0, np.cumsum(np.bincount(integer_ids, minlength=n))]
    theta = theta.copy()
    s = s0.copy()
    mean = b.copy()
    mass = 1.0
    labels = np.zeros(n, dtype=np.int64)
    atoms = b[None, :].copy()
    rng = np.random.default_rng(seed)
    parameters = np.empty((size, n, 6))
    variances = np.empty(size)
    covariances = np.empty((size, 6, 6))
    means = np.empty((size, dimension))
    base_covariances = np.empty((size, dimension, dimension))
    masses = np.empty(size)
    assignments = np.empty((size, n), dtype=np.int64)
    counts = np.empty(size, dtype=np.int64)
    likelihood = np.empty(size)
    atom_draws: list[FloatArray] = []
    accepted = np.zeros((n, 3))
    record = 0
    for iteration in range(1, iterations + 1):
        try:
            vp = anovaddp_variance_posterior(
                theta, t, y, integer_ids, alpha0=a0, beta0=b0, mode=variance_mode
            )
            gamma = float(rng.gamma(vp.shape))
            variance = vp.scale / gamma if gamma > 0 else np.inf
            if not np.isfinite(variance) or variance <= 0:
                raise ArithmeticError("observation variance draw is not finite and positive")
            for i in range(n):
                prior = np.r_[2.0, x[i] @ atoms[labels[i]].reshape(x.shape[1], 5)]
                section = slice(starts[i], starts[i + 1])
                step = anovaddp_subject_update(
                    theta[i],
                    ts[section],
                    ys[section],
                    prior_mean=prior,
                    prior_covariance=s,
                    variance=variance,
                    seed=int(rng.integers(2**63)),
                )
                theta[i] = step.parameters
                accepted[i] += step.accepted
            regression = s[1:, 0] / s[0, 0]
            conditional = theta[:, 1:] - (theta[:, 0] - 2)[:, None] * regression
            conditional_covariance = s[1:, 1:] - np.outer(s[1:, 0], s[0, 1:]) / s[0, 0]
            cluster = anovaddp_cluster_sweep(
                conditional,
                x,
                labels,
                atoms,
                residual_covariance=conditional_covariance,
                base_mean=mean,
                base_covariance=c,
                concentration=mass,
                seed=int(rng.integers(2**63)),
            )
            labels, atoms = cluster.labels, cluster.atoms
            hyper = anovaddp_hyperparameter_update(
                theta,
                x,
                labels,
                atoms,
                base_covariance=c,
                base_prior=b,
                covariance_prior=s0,
                covariance_df=covariance_df,
                concentration=mass,
                concentration_shape=am,
                concentration_rate=bm,
                seed=int(rng.integers(2**63)),
            )
            mean, c, s, mass = (
                hyper.base_mean,
                hyper.base_covariance,
                hyper.residual_covariance,
                hyper.concentration,
            )
            if record < size and iteration == saved[record]:
                parameters[record] = theta
                variances[record] = variance
                covariances[record] = s
                means[record] = mean
                base_covariances[record] = c
                masses[record] = mass
                assignments[record] = labels
                counts[record] = atoms.shape[0]
                atom_draws.append(atoms)
                likelihood[record] = sum(
                    float(
                        anovaddp_loglikelihood(
                            theta[i],
                            ts[starts[i] : starts[i + 1]],
                            ys[starts[i] : starts[i + 1]],
                            variance,
                            normalized=True,
                        )
                    )
                    for i in range(n)
                )
                record += 1
        except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
            raise ArithmeticError(f"ANOVA DDP iteration {iteration} failed: {exc}") from exc
    return AnovaDDPFit(
        _freeze(parameters),
        _freeze(variances),
        _freeze(covariances),
        _freeze(means),
        _freeze(base_covariances),
        _freeze(masses),
        np.frombuffer(assignments.tobytes(), dtype=np.int64).reshape(size, n),
        tuple(atom_draws),
        np.frombuffer(counts.tobytes(), dtype=np.int64),
        _freeze(likelihood),
        _freeze(accepted / iterations),
        np.frombuffer(saved.tobytes(), dtype=np.int64),
        variance_mode,
    )
