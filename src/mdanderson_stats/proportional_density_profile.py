"""Common-censoring profile-likelihood inference for the density tilt."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import brentq
from scipy.special import erfinv, expit, ndtr

from ._validation import scalar
from .proportional_density import proportional_density

_PROFILE_ITERATIONS = 80
_ROOT_BRACKET_ITERATIONS = 64
_ROOT_ITERATIONS = 100
_FIT_VECTOR_PASSES = 11_000
_PROFILE_VECTOR_PASSES = 5
_DEFAULT_MAX_WORK = 500_000_000


@dataclass(frozen=True)
class ProportionalDensityProfileResult:
    """Profile test for a fixed beta and optional inverted beta interval."""

    beta_null: float
    alpha_at_null: float
    likelihood_ratio: float
    pvalue: float
    beta_estimate: float
    confidence: float | None
    beta_interval: tuple[float, float] | None
    profile_evaluations: int
    work_bound: int


def proportional_density_profile(
    time: ArrayLike,
    event: ArrayLike,
    treatment: ArrayLike,
    *,
    beta_null: float,
    equal_censoring: bool,
    confidence: float = 0.95,
    interval: bool = True,
    max_work: int = _DEFAULT_MAX_WORK,
) -> ProportionalDensityProfileResult:
    """Test a fixed treatment-time slope under the common-censoring model.

    The conditional likelihood ratio profiles the intercept ``alpha`` at the
    requested ``beta_null``. The paper gives chi-square-one calibration when
    censoring distributions are common. The caller must explicitly assert that
    assumption with ``equal_censoring=True``. Setting ``interval=True`` inverts
    this same test to obtain a beta interval; the interval is a Python extension
    of the paper's test, not a native interval procedure.

    ``max_work`` bounds a conservative count of event-vector operations before
    fitting. Confidence limits are found by bounded bracketing and root finding;
    failure to represent or bracket a finite limit raises rather than returning
    a fabricated endpoint. ``alpha_star`` is a derived curve-normalization
    parameter and is not the nuisance intercept profiled here.
    """
    if not isinstance(equal_censoring, (bool, np.bool_)):
        raise ValueError("equal_censoring must be explicitly True or False")
    if not equal_censoring:
        raise ValueError("profile-likelihood beta inference requires equal_censoring=True")
    if not isinstance(interval, (bool, np.bool_)):
        raise ValueError("interval must be boolean")
    if (
        isinstance(max_work, (bool, np.bool_))
        or not isinstance(max_work, (int, np.integer))
        or max_work < 1
    ):
        raise ValueError("max_work must be a positive integer")
    beta0 = scalar(beta_null, "beta_null")
    level: float | None = scalar(confidence, "confidence") if interval else None
    if interval:
        assert level is not None
        if not 0 < level < 1:
            raise ValueError("confidence must be in (0,1)")

    raw = tuple(np.asarray(value) for value in (time, event, treatment))
    if any(np.iscomplexobj(value) for value in raw):
        raise ValueError("time, event and treatment must be real")
    if (
        raw[0].ndim != 1
        or not 4 <= raw[0].size <= 1_000_000
        or any(value.shape != raw[0].shape for value in raw[1:])
    ):
        raise ValueError("require equal-length vectors with 4 to 1,000,000 observations")
    t = np.asarray(raw[0], dtype=np.float64)
    d = np.asarray(raw[1], dtype=np.float64)
    z = np.asarray(raw[2], dtype=np.float64)
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(d)) and np.all(np.isfinite(z))):
        raise ValueError("time, event and treatment must be finite")
    if np.any(t < 0) or np.any((d != 0) & (d != 1)) or np.any((z != 0) & (z != 1)):
        raise ValueError("time must be nonnegative; event and treatment must be 0 or 1")
    failures = d == 1
    x = t[failures]
    y = z[failures]
    m0 = int(np.count_nonzero(y == 0))
    m1 = int(np.count_nonzero(y == 1))
    if m0 == 0 or m1 == 0:
        raise ValueError("each arm must have observed failures")
    event_count = int(x.size)
    # The unrestricted 2D Newton fit has <=100 iterations. A profile evaluation
    # has <=80 intercept iterations. Each CI endpoint has <=64 bracketing and
    # <=100 root evaluations. Include input scans as well as event-vector work.
    # brentq can evaluate both endpoints in addition to maxiter interior points;
    # the bracket confirmation adds one call after its bounded expansion.
    profile_calls = 2 + (
        2 * (_ROOT_BRACKET_ITERATIONS + 1 + _ROOT_ITERATIONS + 2) if interval else 0
    )
    record_passes = 3 * (32 + t.size.bit_length()) + 16
    bound = int(
        t.size * record_passes
        + event_count
        * (_FIT_VECTOR_PASSES + profile_calls * _PROFILE_ITERATIONS * _PROFILE_VECTOR_PASSES)
    )
    if bound > int(max_work):
        raise ValueError(
            f"requested profile inference requires up to {bound:,} work units, "
            f"exceeding max_work={int(max_work):,}"
        )

    # Reuse the validated unrestricted fit and its stable centered/scaled solver.
    fit = proportional_density(t, d, z, equal_censoring=True)
    lo, hi = float(np.min(x)), float(np.max(x))
    if lo == hi:
        raise ValueError("failure times must vary to identify beta")
    center = lo / 2 + hi / 2
    scale = max(hi - center, center - lo)
    centered = (x - center) / scale
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        slope_hat = float(fit.beta * scale)
        slope0 = float(beta0 * scale)
    if not (np.isfinite(slope_hat) and np.isfinite(slope0)):
        raise ArithmeticError("scaled beta is not representable; rescale the time units")

    offset = float(np.log(m1 / m0))
    target = float(m1)
    total = float(event_count)
    profile_evaluations = 0

    def profile(slope: float) -> tuple[float, float]:
        """Return profiled centered intercept and conditional log likelihood."""
        nonlocal profile_evaluations
        profile_evaluations += 1
        base = offset + slope * centered
        if np.any(~np.isfinite(base)):
            raise ArithmeticError("profile linear predictor is not representable")
        # These values bracket the root because every linear predictor lies
        # between min(base) and max(base).
        logit_fraction = float(np.log(m1) - np.log(m0))
        lower = logit_fraction - float(np.max(base))
        upper = logit_fraction - float(np.min(base))
        if not (np.isfinite(lower) and np.isfinite(upper)):
            raise ArithmeticError("profile intercept bracket is not representable")
        intercept = lower / 2 + upper / 2
        tolerance = 8 * np.finfo(float).eps * max(1.0, total)
        for _ in range(_PROFILE_ITERATIONS):
            eta = base + intercept
            p = expit(eta)
            score = float(np.sum(p) - target)
            if abs(score) <= tolerance:
                break
            if score < 0:
                lower = intercept
            else:
                upper = intercept
            information = float(np.sum(p * (1 - p)))
            candidate = intercept - score / information if information > 0 else np.nan
            if not np.isfinite(candidate) or not lower < candidate < upper:
                candidate = lower / 2 + upper / 2
            intercept = candidate
            if upper - lower <= 4 * np.finfo(float).eps * max(1.0, abs(intercept)):
                intercept = lower / 2 + upper / 2
                break
        else:
            raise ArithmeticError("profile intercept did not converge in 80 iterations")
        eta = base + intercept
        signed = 1 - 2 * y
        log_likelihood = -float(np.sum(np.logaddexp(0.0, signed * eta)))
        if not np.isfinite(log_likelihood):
            raise ArithmeticError("profile likelihood is not representable")
        return intercept, log_likelihood

    intercept0, loglik0 = profile(slope0)
    _, loglik_hat = profile(slope_hat)
    lr = 2 * (loglik_hat - loglik0)
    roundoff = 64 * np.finfo(float).eps * max(1.0, event_count, abs(loglik_hat))
    if lr < -roundoff:
        raise ArithmeticError("profile likelihood exceeds the unrestricted fit")
    lr = max(0.0, float(lr))
    pvalue = float(2 * ndtr(-np.sqrt(lr)))

    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        alpha_null = float(intercept0 - slope0 * (center / scale))
    if not np.isfinite(alpha_null):
        raise ArithmeticError("profiled alpha is not representable in original time units")
    beta_interval: tuple[float, float] | None = None
    if interval:
        assert level is not None
        # Chi-square(1) cutoff: z_(1+c)/2**2 = 2*erfinv(c)**2. This form
        # remains representable in both tails, including c near zero and one.
        cutoff = float(2 * erfinv(float(level)) ** 2)
        if cutoff <= roundoff:
            raise ArithmeticError(
                "confidence cutoff is below the profile likelihood's floating-point resolution"
            )

        def signed_profile_lr(slope: float) -> float:
            _, ll = profile(slope)
            value = 2 * (loglik_hat - ll) - cutoff
            if not np.isfinite(value):
                raise ArithmeticError("profile interval root is not representable")
            return float(value)

        endpoints: list[float] = []
        for direction in (-1.0, 1.0):
            step = max(0.01, abs(slope_hat) * 0.1, 1.0)
            edge = slope_hat + direction * step
            for _ in range(_ROOT_BRACKET_ITERATIONS):
                if not np.isfinite(edge):
                    break
                if signed_profile_lr(edge) >= 0:
                    break
                step *= 2
                edge = slope_hat + direction * step
            else:
                edge = np.nan
            if not np.isfinite(edge) or signed_profile_lr(edge) < 0:
                raise ArithmeticError("could not bracket a finite beta confidence limit")
            lo_root, hi_root = sorted((edge, slope_hat))
            try:
                root = float(
                    brentq(
                        signed_profile_lr,
                        lo_root,
                        hi_root,
                        xtol=4 * np.finfo(float).eps,
                        rtol=8 * np.finfo(float).eps,
                        maxiter=_ROOT_ITERATIONS,
                    )
                )
            except (ValueError, RuntimeError) as exc:
                raise ArithmeticError("beta confidence-limit root did not converge") from exc
            residual = abs(signed_profile_lr(root))
            residual_tolerance = 1e-10 * max(1.0, event_count, cutoff)
            if residual > residual_tolerance:
                raise ArithmeticError("beta confidence-limit likelihood residual is too large")
            endpoints.append(root / scale)
        if not np.all(np.isfinite(endpoints)):
            raise ArithmeticError("beta confidence limits are not representable")
        beta_interval = (float(endpoints[0]), float(endpoints[1]))

    return ProportionalDensityProfileResult(
        beta0,
        alpha_null,
        lr,
        pvalue,
        fit.beta,
        float(level) if level is not None else None,
        beta_interval,
        profile_evaluations,
        bound,
    )
