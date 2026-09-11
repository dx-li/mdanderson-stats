"""Design-prior operating characteristics for Bayesian success decisions."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import quad
from scipy.special import betainc, betaincc, ndtr, ndtri
from scipy.stats import betabinom, binom

from ._validation import finite, scalar
from .beta_binomial import BetaBinomialPosterior, _owned
from .beta_comparison import BetaDifferenceComparison, compare_beta_difference


@dataclass(frozen=True)
class SuccessOperatingCharacteristics:
    true_positive: float
    false_positive: float
    true_negative: float
    false_negative: float
    bayesian_power: float
    bayesian_conditional_power: float | None
    bayesian_type1_error: float | None
    incorrect_decision_probability: float | None
    false_omission_rate: float | None
    frequentist_type1_error: float


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0 else None


def _result(
    tp: float, fp: float, tn: float, fn: float, alpha: float
) -> SuccessOperatingCharacteristics:
    mass = np.array([tp, fp, tn, fn])
    if np.any(mass < 0) or not np.all(np.isfinite(mass)) or abs(mass.sum() - 1) > 1e-8:
        raise ArithmeticError("decision-state probabilities do not form a valid partition")
    return SuccessOperatingCharacteristics(
        tp,
        fp,
        tn,
        fn,
        tp + fp,
        _ratio(tp, tp + fn),
        _ratio(fp, fp + tn),
        _ratio(fp, tp + fp),
        _ratio(fn, fn + tn),
        alpha,
    )


def _cutoff(value: float) -> float:
    value = scalar(value, "cutoff")
    if not 0 <= value <= 1:
        raise ValueError("cutoff must be in [0,1]")
    return value


def _direction(value: str) -> int:
    if value not in ("greater", "less"):
        raise ValueError("direction must be greater or less")
    return 1 if value == "greater" else -1


def binary_success_oc(
    n: int,
    cutoff: float,
    *,
    margin: float,
    design_prior: tuple[float, float] = (1.0, 1.0),
    analysis_prior: tuple[float, float] = (1.0, 1.0),
    direction: str = "greater",
    null_rate: float | None = None,
) -> SuccessOperatingCharacteristics:
    """Exact single-arm beta-binomial success OCs, with strict posterior > cutoff.

    The design prior generates truth and observations. The analysis prior only
    determines which response counts declare success. Undefined conditional
    metrics (zero-probability conditioning events) are returned as None.
    """
    size = scalar(n, "n")
    if size != np.floor(size) or not 1 <= size <= 100000:
        raise ValueError("n must be an integer in [1,100000]")
    size = int(size)
    c, sign = _cutoff(cutoff), _direction(direction)
    delta = scalar(margin, "margin")
    null = delta if null_rate is None else scalar(null_rate, "null_rate")
    if not 0 <= delta <= 1 or not 0 <= null <= 1:
        raise ValueError("margin and null_rate must be in [0,1]")
    design, analysis = (
        finite(design_prior, "design_prior"),
        finite(analysis_prior, "analysis_prior"),
    )
    if (
        design.shape != (2,)
        or analysis.shape != (2,)
        or np.any(design <= 0)
        or np.any(analysis <= 0)
    ):
        raise ValueError("each prior must contain two positive beta shape parameters")
    x = np.arange(size + 1)
    tail = betaincc if sign == 1 else betainc
    pa = tail(analysis[0] + x, analysis[1] + size - x, delta)
    effective = tail(design[0] + x, design[1] + size - x, delta)
    opposite = betainc if sign == 1 else betaincc
    ineffective = opposite(design[0] + x, design[1] + size - x, delta)
    predictive = betabinom.pmf(x, size, *design)
    success = pa > c
    tp, fp = (
        np.sum(predictive[success] * effective[success]),
        np.sum(predictive[success] * ineffective[success]),
    )
    fn, tn = (
        np.sum(predictive[~success] * effective[~success]),
        np.sum(predictive[~success] * ineffective[~success]),
    )
    alpha = binom.pmf(x[success], size, null).sum()
    return _result(float(tp), float(fp), float(tn), float(fn), float(alpha))


def _lower_quadrant(a: float, b: float, rho: float) -> float:
    """Direct lower/lower probability, splitting to resolve the density and CDF."""
    sd = np.sqrt((1 - rho) * (1 + rho))
    if sd <= 0:
        raise ArithmeticError("normal correlation is numerically singular")
    # Beyond 40 standard deviations normal density is below float64 range.
    # Finite bounds also prevent huge conditional-CDF crossing points from
    # making adaptive quadrature miss the normal density near zero.
    if a <= -40:
        return 0.0
    a = min(a, 40.0)
    points = sorted({-40.0, a, *(p for p in (0.0, b / rho) if -40 < p < a)})
    total = 0.0
    for left, right in zip(points[:-1], points[1:], strict=True):
        out = quad(
            lambda x: np.exp(-x * x / 2) / np.sqrt(2 * np.pi) * ndtr((b - rho * x) / sd),
            left,
            right,
            epsabs=2e-12,
            epsrel=2e-11,
            limit=200,
            full_output=1,
        )
        if len(out) != 3 or out[1] > 1e-9:
            raise ArithmeticError("normal decision probability integration did not converge")
        total += out[0]
    return float(total)


def normal_success_oc(
    cutoff: float,
    *,
    standard_error: ArrayLike,
    design_mean: ArrayLike = 0.0,
    design_sd: ArrayLike = 1.0,
    analysis_mean: ArrayLike = 0.0,
    analysis_sd: ArrayLike = 1.0,
    margin: float = 0.0,
    direction: str = "greater",
    null_mean: ArrayLike | None = None,
) -> SuccessOperatingCharacteristics:
    """Known-variance normal success OCs for one effect or two independent arms.

    Supply a scalar standard error for a single mean or log hazard ratio, or a
    two-element vector [treatment, control] for a difference of arm means. Priors
    broadcast across arms. Two-arm frequentist OCs require explicit null means;
    a scalar null for a one-effect model defaults to the clinical margin.
    """
    c, sign = _cutoff(cutoff), _direction(direction)
    delta = scalar(margin, "margin")
    values = [
        finite(v, name)
        for v, name in (
            (standard_error, "standard_error"),
            (design_mean, "design_mean"),
            (design_sd, "design_sd"),
            (analysis_mean, "analysis_mean"),
            (analysis_sd, "analysis_sd"),
        )
    ]
    se, dm, ds, am, ass = [np.atleast_1d(v) for v in np.broadcast_arrays(*values)]
    if se.shape not in ((1,), (2,)) or np.any(se <= 0) or np.any(ds <= 0) or np.any(ass <= 0):
        raise ValueError("require one or two independent arms and positive standard deviations")
    if null_mean is None and len(se) == 2:
        raise ValueError("two-arm frequentist evaluation requires null_mean for both arms")
    null = np.broadcast_to(finite(delta if null_mean is None else null_mean, "null_mean"), se.shape)
    coefficients = sign * np.array([1.0] if len(se) == 1 else [1.0, -1.0])
    # Scale-safe posterior weight and variance; no inverse variance overflow.
    scale = np.hypot(ass, se)
    weight = (ass / scale) ** 2
    post_sd = ass * (se / scale)
    true_sd = float(np.hypot.reduce(ds))
    score_sd = float(np.hypot.reduce(weight * np.hypot(ds, se)))
    noise_sd = float(np.hypot.reduce(weight * se))
    posterior_sd = float(np.hypot.reduce(post_sd))
    true_mean = float(coefficients @ dm)
    score_mean = float(coefficients @ (weight * dm + (1 - weight) * am))
    null_score = float(coefficients @ (weight * null + (1 - weight) * am))
    rho = float(np.sum((weight * ds / score_sd) * (ds / true_sd)))
    if not 0 < rho < 1 or not np.isfinite(score_sd) or noise_sd <= 0:
        raise ArithmeticError("normal model scale cannot be resolved")
    a = (sign * delta - true_mean) / true_sd
    if c in (0.0, 1.0):
        effective, ineffective = float(ndtr(-a)), float(ndtr(a))
        return (
            _result(effective, ineffective, 0.0, 0.0, 1.0)
            if c == 0
            else _result(0.0, 0.0, ineffective, effective, 0.0)
        )
    boundary = sign * delta + float(ndtri(c)) * posterior_sd
    b = (boundary - score_mean) / score_sd
    # Reflect variables to evaluate each quadrant directly, preserving small tails.
    tn = _lower_quadrant(a, b, rho)
    tp = _lower_quadrant(-a, -b, rho)
    fp = _lower_quadrant(a, -b, -rho)
    fn = _lower_quadrant(-a, b, -rho)
    alpha = float(ndtr((null_score - boundary) / noise_sd))
    return _result(tp, fp, tn, fn, alpha)


@dataclass(frozen=True)
class SuccessCalibration:
    cutoff: float
    target: float
    operating_characteristics: SuccessOperatingCharacteristics
    candidates_evaluated: int


def calibrate_success_cutoff(
    evaluate: Callable[[float], SuccessOperatingCharacteristics],
    target: float,
    candidates: ArrayLike,
) -> SuccessCalibration:
    """Smallest supplied cutoff meeting the target probability of incorrect decision.

    Search actual candidate cutoffs without assuming a continuous error function.
    Zero-success designs have undefined PID and never count as feasible.
    """
    target = scalar(target, "target")
    grid = finite(candidates, "candidates")
    if not 0 < target < 1 or grid.ndim != 1 or not len(grid) or np.any((grid < 0) | (grid > 1)):
        raise ValueError("require target in (0,1) and a nonempty cutoff vector in [0,1]")
    for i, c in enumerate(np.unique(grid)):
        result = evaluate(float(c))
        pid = result.incorrect_decision_probability
        if pid is not None and pid <= target:
            return SuccessCalibration(float(c), target, result, i + 1)
    raise ValueError(
        "no candidate cutoff achieves the PID target with positive success probability"
    )


def binary_two_arm_success_oc(
    n_treatment: int,
    n_control: int,
    cutoff: float,
    *,
    design_treatment: tuple[float, float] = (1.0, 1.0),
    design_control: tuple[float, float] = (1.0, 1.0),
    analysis_treatment: tuple[float, float] = (1.0, 1.0),
    analysis_control: tuple[float, float] = (1.0, 1.0),
    null_rate: float = 0.5,
    margin: float = 0.0,
    null_treatment_rate: float | None = None,
    direction: str = "greater",
    absolute_tolerance: float = 1e-10,
) -> SuccessOperatingCharacteristics:
    """Two-arm binary success OCs with an arbitrary risk-difference margin.

    For many cutoffs, use prepare_binary_two_arm_success once and reuse its
    evaluate method, avoiding repeated posterior quadrature.
    """
    table = prepare_binary_two_arm_success(
        n_treatment,
        n_control,
        design_treatment=design_treatment,
        design_control=design_control,
        analysis_treatment=analysis_treatment,
        analysis_control=analysis_control,
        null_rate=null_rate,
        margin=margin,
        null_treatment_rate=null_treatment_rate,
        direction=direction,
        absolute_tolerance=absolute_tolerance,
    )
    return table.evaluate(cutoff)


@dataclass(frozen=True)
class BinarySuccessTable:
    """Owned read-only response-count arrays reusable across cutoff searches."""

    posterior_probability: np.ndarray
    posterior_error: np.ndarray
    effective_mass: np.ndarray
    ineffective_mass: np.ndarray
    null_mass: np.ndarray

    def evaluate(self, cutoff: float) -> SuccessOperatingCharacteristics:
        c = _cutoff(cutoff)
        pa = self.posterior_probability
        if 0 < c < 1 and np.any((self.posterior_error > 0) & (abs(pa - c) <= self.posterior_error)):
            raise ArithmeticError("cutoff cannot be distinguished from posterior quadrature error")
        success = (pa > c) | ((c == 0) & (self.posterior_error > 0))
        tp, fp = (
            float(self.effective_mass[success].sum()),
            float(self.ineffective_mass[success].sum()),
        )
        tn, fn = (
            float(self.ineffective_mass[~success].sum()),
            float(self.effective_mass[~success].sum()),
        )
        return _result(tp, fp, tn, fn, float(self.null_mass[success].sum()))


def prepare_binary_two_arm_success(
    n_treatment: int,
    n_control: int,
    *,
    design_treatment: tuple[float, float] = (1.0, 1.0),
    design_control: tuple[float, float] = (1.0, 1.0),
    analysis_treatment: tuple[float, float] = (1.0, 1.0),
    analysis_control: tuple[float, float] = (1.0, 1.0),
    null_rate: float = 0.5,
    margin: float = 0.0,
    null_treatment_rate: float | None = None,
    direction: str = "greater",
    absolute_tolerance: float = 1e-10,
) -> BinarySuccessTable:
    """Prepare quadrature and predictive masses once for repeated decisions.

    null_rate is the control rate. Treatment defaults to the same rate for
    compatibility; specify null_treatment_rate to evaluate a margin-boundary null.
    """
    nt, nc = scalar(n_treatment, "n_treatment"), scalar(n_control, "n_control")
    if any(n != np.floor(n) or not 1 <= n <= 1000 for n in (nt, nc)):
        raise ValueError("arm sizes must be integers in [1,1000]")
    nt, nc = int(nt), int(nc)
    if (nt + 1) * (nc + 1) > 40000:
        raise ValueError("two-arm enumeration is limited to 40000 response-count pairs")
    sign = _direction(direction)
    delta = scalar(margin, "margin")
    if not -1 <= delta <= 1:
        raise ValueError("margin must be in [-1,1]")
    null = scalar(null_rate, "null_rate")
    if not 0 <= null <= 1:
        raise ValueError("null_rate must be in [0,1]")
    null_t = (
        null if null_treatment_rate is None else scalar(null_treatment_rate, "null_treatment_rate")
    )
    if not 0 <= null_t <= 1:
        raise ValueError("null_treatment_rate must be in [0,1]")
    priors = [
        finite(v, name)
        for v, name in (
            (design_treatment, "design_treatment"),
            (design_control, "design_control"),
            (analysis_treatment, "analysis_treatment"),
            (analysis_control, "analysis_control"),
        )
    ]
    if any(p.shape != (2,) or np.any(p <= 0) for p in priors):
        raise ValueError("each prior must contain two positive beta shape parameters")
    dt, dc, at, ac = priors
    t, control = np.arange(nt + 1)[:, None], np.arange(nc + 1)[None, :]

    def compare(pt: np.ndarray, pc: np.ndarray) -> BetaDifferenceComparison:
        return compare_beta_difference(
            BetaBinomialPosterior(pc[0] + control, pc[1] + nc - control),
            BetaBinomialPosterior(pt[0] + t, pt[1] + nt - t),
            margin=delta,
            absolute_tolerance=absolute_tolerance,
        )

    analysis = compare(at, ac)
    design = analysis if np.array_equal(at, dt) and np.array_equal(ac, dc) else compare(dt, dc)
    pa = analysis.above_margin if sign == 1 else analysis.below_margin
    effective = design.above_margin if sign == 1 else design.below_margin
    ineffective = design.below_margin if sign == 1 else design.above_margin
    predictive = betabinom.pmf(t, nt, *dt) * betabinom.pmf(control, nc, *dc)
    null_mass = binom.pmf(t, nt, null_t) * binom.pmf(control, nc, null)
    return BinarySuccessTable(
        _owned(pa),
        _owned(analysis.absolute_error),
        _owned(predictive * effective),
        _owned(predictive * ineffective),
        _owned(null_mass),
    )
