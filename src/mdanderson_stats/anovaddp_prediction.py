"""New-patient and study prediction workflows supplied with ANOVA DDP."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import expit

from ._cdflib import _freeze
from ._validation import FloatArray, count, finite, scalar
from .anovaddp import anovaddp_curve
from .anovaddp_clusters import _atom, _inputs
from .anovaddp_mcmc import AnovaDDPFit


@dataclass(frozen=True)
class AnovaDDPNewAtom:
    coefficients: FloatArray
    cluster: int
    weights: FloatArray


def anovaddp_new_atom(
    conditional_parameters: ArrayLike,
    design: ArrayLike,
    labels: ArrayLike,
    *,
    residual_covariance: ArrayLike,
    base_mean: ArrayLike,
    base_covariance: ArrayLike,
    concentration: float,
    seed: int | None = None,
) -> AnovaDDPNewAtom:
    """Native Newpatient: count-weighted atom posterior or Gaussian base draw.

    Existing-cluster coefficients are freshly drawn conditional on its subjects,
    as in the source. cluster=-1 denotes the new-cluster/base option, which is
    the final entry in weights. This does not alter training allocations.
    """
    y, x, m, s, _, c = _inputs(
        conditional_parameters, design, residual_covariance, base_mean, base_covariance
    )
    if np.iscomplexobj(labels):
        raise ValueError("labels must be real")
    raw = count(labels, "labels")
    if raw.shape != (y.shape[0],) or np.any(raw >= y.shape[0]):
        raise ValueError("labels must be aligned contiguous cluster IDs")
    lab = raw.astype(np.int64)
    sizes = np.bincount(lab)
    if np.any(sizes == 0):
        raise ValueError("labels must be contiguous occupied cluster IDs")
    mass = scalar(concentration, "concentration")
    if mass <= 0 or (
        seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0)
    ):
        raise ValueError("positive concentration and nonnegative integer seed required")
    weights = np.r_[sizes.astype(float), mass]
    weights /= weights.max()
    weights /= weights.sum()
    rng = np.random.default_rng(seed)
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1
    selected = int(np.searchsorted(cumulative, rng.random(), side="right"))
    if selected == sizes.size:
        mean, covariance = m, c
        cluster = -1
    else:
        mask = lab == selected
        posterior = _atom(y[mask], x[mask], s, m, c)
        mean, covariance = posterior.mean, posterior.covariance
        cluster = selected
    coefficients = mean + np.linalg.cholesky(covariance) @ rng.standard_normal(m.size)
    if not np.all(np.isfinite(coefficients)):
        raise ArithmeticError("new atom draw exceeds floating-point range")
    return AnovaDDPNewAtom(_freeze(coefficients), cluster, _freeze(weights))


def anovaddp_baseline_curves(coefficients: ArrayLike, time: ArrayLike) -> FloatArray:
    """Native ten-row Baseline transformations for seven covariate blocks.

    Rows: study 3, study-2 component, study 1, three no-GM components, three
    with-GM components, GM only. These nonlinear component curves are not
    differences of fitted response curves. fittare knot repair is applied.
    """
    if np.iscomplexobj(coefficients):
        raise ValueError("coefficients must be real")
    a = finite(coefficients, "coefficients")
    if a.shape != (35,):
        raise ValueError("Baseline requires seven blocks of five coefficients")
    blocks = a.reshape(7, 5)
    means = np.empty((10, 6))
    means[:, 0] = [2, 0, 2, 0, 0, 0, 0, 0, 0, 0]
    means[:, 1:] = np.vstack(
        (
            blocks[0],
            blocks[6],
            blocks[0] + blocks[5],
            blocks[1:4],
            blocks[1:4] + blocks[4],
            blocks[4],
        )
    )
    return anovaddp_curve(means, time, repair_order=True)


@dataclass(frozen=True)
class AnovaDDPPrediction:
    time: FloatArray
    common_effect: FloatArray
    study3: FloatArray
    nadir: FloatArray
    baseline_mean: FloatArray
    baseline_second_moment: FloatArray
    prediction_mean: FloatArray
    prediction_second_moment: FloatArray
    baseline_draws: FloatArray
    prediction_draws: FloatArray
    atom_draws: FloatArray
    selected_cluster: NDArray[np.int64]


def predict_anovaddp(
    fit: AnovaDDPFit,
    training_design: ArrayLike,
    prediction_design: ArrayLike,
    *,
    time: ArrayLike = tuple(range(-1, 31)),
    seed: int | None = None,
) -> AnovaDDPPrediction:
    """Replay all seven native predictive outputs from retained posterior states.

    Uses the source's seven-covariate study layout, at least three prediction
    rows, and one shared new atom per state. Curves contain latent subject
    variation but no observation-error draw. Nadir draws are independent of
    prediction curves and cover the first three design rows, as in NADIRSTEP.
    """
    if not isinstance(fit, AnovaDDPFit):
        raise ValueError("fit must be an AnovaDDPFit")
    if any(np.iscomplexobj(a) for a in (training_design, prediction_design, time)):
        raise ValueError("designs and time must be real")
    train, x, t = (
        finite(a, n)
        for a, n in zip(
            (training_design, prediction_design, time),
            ("training_design", "prediction_design", "time"),
        )
    )
    if fit.parameters.ndim != 3 or fit.parameters.shape[2] != 6 or not fit.parameters.shape[0]:
        raise ValueError("fit requires nonempty six-parameter posterior states")
    draws, subjects = fit.parameters.shape[:2]
    if (
        train.shape != (subjects, 7)
        or x.ndim != 2
        or x.shape[1] != 7
        or not 3 <= x.shape[0] <= 1000
        or np.any(train[:, 0] != 1)
        or np.any(x[:, 0] != 1)
        or t.ndim != 1
        or not t.size
    ):
        raise ValueError(
            "require aligned seven-column designs, intercept one, "
            ">=3 prediction rows and nonempty time"
        )
    expected = (
        (fit.labels, (draws, subjects)),
        (fit.base_mean, (draws, 35)),
        (fit.base_covariance, (draws, 35, 35)),
        (fit.residual_covariance, (draws, 6, 6)),
        (fit.concentration, (draws,)),
    )
    if any(a.shape != shape or not np.all(np.isfinite(a)) for a, shape in expected):
        raise ValueError("posterior state dimensions or values are invalid")
    if draws * (t.size * (12 + x.shape[0]) + 38) > 30_000_000:
        raise ValueError("prediction exceeds 30 million saved values")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
        raise ValueError("seed must be a nonnegative integer or None")
    rng = np.random.default_rng(seed)
    common = np.empty((draws, t.size))
    study = np.empty_like(common)
    nadir = np.empty((draws, 3))
    baseline = np.empty((draws, 10, t.size))
    prediction = np.empty((draws, x.shape[0], t.size))
    atoms = np.empty((draws, 35))
    selected = np.empty(draws, dtype=np.int64)
    for j in range(draws):
        s = fit.residual_covariance[j]
        theta = fit.parameters[j]
        residual = theta[:, 1:] - (theta[:, 0] - 2)[:, None] * (s[1:, 0] / s[0, 0])
        conditional = s[1:, 1:] - np.outer(s[1:, 0], s[0, 1:]) / s[0, 0]
        atom = anovaddp_new_atom(
            residual,
            train,
            fit.labels[j],
            residual_covariance=conditional,
            base_mean=fit.base_mean[j],
            base_covariance=fit.base_covariance[j],
            concentration=float(fit.concentration[j]),
            seed=int(rng.integers(2**63)),
        )
        atoms[j], selected[j] = atom.coefficients, atom.cluster
        mean = np.r_[2.0, atom.coefficients[:5]]
        common[j] = anovaddp_curve(mean, t, repair_order=True)
        chol = np.linalg.cholesky(s)
        study[j] = anovaddp_curve(mean + chol @ rng.standard_normal(6), t, repair_order=True)
        baseline[j] = anovaddp_baseline_curves(atom.coefficients, t)
        means = np.column_stack((np.full(x.shape[0], 2.0), x @ atom.coefficients.reshape(7, 5)))
        patients = means + rng.standard_normal((x.shape[0], 6)) @ chol.T
        prediction[j] = anovaddp_curve(patients, t, repair_order=True)
        nadir_patients = means[:3] + rng.standard_normal((3, 6)) @ chol.T
        nadir[j] = nadir_patients[:, 1] + nadir_patients[:, 2] * expit(-2.0)
    with np.errstate(over="ignore"):
        baseline_second = np.mean(baseline**2, axis=0)
        prediction_second = np.mean(prediction**2, axis=0)
    if not all(np.all(np.isfinite(a)) for a in (nadir, baseline_second, prediction_second)):
        raise ArithmeticError("predictive summaries exceed floating-point range")
    return AnovaDDPPrediction(
        _freeze(t),
        _freeze(common),
        _freeze(study),
        _freeze(nadir),
        _freeze(baseline.mean(axis=0)),
        _freeze(baseline_second),
        _freeze(prediction.mean(axis=0)),
        _freeze(prediction_second),
        _freeze(baseline),
        _freeze(prediction),
        _freeze(atoms),
        np.frombuffer(selected.tobytes(), dtype=np.int64),
    )
