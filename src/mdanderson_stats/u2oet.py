"""U2OET PDS/GCR ordinal model, from Thall, Nguyen and Zinner (2017)."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from ._cdflib import _freeze
from ._validation import FloatArray, finite


def _real(value: ArrayLike, name: str) -> FloatArray:
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    return finite(value, name)


def u2oet_standardize(doses: ArrayLike, power: float = 1.0) -> FloatArray:
    """Paper equation (1), on a strictly increasing positive 2–5 dose grid."""
    d = _real(doses, "doses")
    p = _real(power, "power")
    if d.ndim != 1 or not 2 <= d.size <= 5 or np.any(d <= 0) or np.any(np.diff(d) <= 0):
        raise ValueError("doses must be 2–5 strictly increasing positive values")
    if p.ndim or p <= 0:
        raise ValueError("power must be a positive scalar")
    # Scaling first prevents overflow of the mean or endpoint difference.
    scaled = d / d[-1]
    if scaled[0] == 0:
        raise ValueError("dose ratios exceed floating-point range")
    span = 1 - scaled[0]
    position = (scaled - scaled[0]) / span
    answer = (scaled[0] + position ** float(p) * span) / scaled.mean()
    return _freeze(answer)


@dataclass(frozen=True)
class U2OETMarginal:
    """One outcome: threshold intercepts, positive slopes, PDS powers and link.

    Slopes have shape (number of categories minus one, 2). The interaction
    coefficient is shared across all thresholds, as in the paper's CMI model.
    """

    intercepts: ArrayLike
    slopes: ArrayLike
    powers: ArrayLike = (1.0, 1.0)
    link: float = 1.0
    interaction: float = 0.0

    def __post_init__(self) -> None:
        a = _real(self.intercepts, "intercepts")
        b = _real(self.slopes, "slopes")
        p = _real(self.powers, "powers")
        phi = _real(self.link, "link")
        interaction = _real(self.interaction, "interaction")
        if a.ndim != 1 or not 1 <= a.size <= 3 or b.shape != (a.size, 2):
            raise ValueError("require 1–3 intercepts and a matching threshold-by-two slope matrix")
        if np.any(b <= 0) or p.shape != (2,) or np.any(p <= 0):
            raise ValueError("slopes and both PDS powers must be positive")
        if phi.ndim or phi <= 0 or interaction.ndim:
            raise ValueError("link must be a positive scalar and interaction a finite scalar")
        for name, value in (("intercepts", a), ("slopes", b), ("powers", p)):
            object.__setattr__(self, name, _freeze(value))
        object.__setattr__(self, "link", float(phi))
        object.__setattr__(self, "interaction", float(interaction))


def _ordinal(eta: FloatArray, phi: float) -> FloatArray:
    """Log category probabilities without subtracting nearly equal CDFs."""
    z = eta + np.log(phi)
    if not np.all(np.isfinite(z)):
        raise ArithmeticError("link predictor exceeds floating-point range")
    log_softplus = np.empty_like(z)
    small = z < -36
    log_softplus[small] = z[small]
    log_softplus[~small] = np.log(np.logaddexp(0, z[~small]))
    log_hazard = log_softplus - np.log(phi)
    with np.errstate(over="ignore", under="ignore", divide="ignore"):
        hazard = np.exp(log_hazard)
        log_gamma = np.log(-np.expm1(-hazard))
    # log(1-exp(-h)) = log(h) + O(h), even when h itself underflows.
    tiny = log_hazard < -36
    log_gamma[tiny] = log_hazard[tiny]
    survival = np.concatenate((np.zeros((*eta.shape[:-1], 1)), np.cumsum(log_gamma, -1)), -1)
    return np.concatenate((survival[..., :-1] - hazard, survival[..., -1:]), -1)


def _marginal(
    doses1: ArrayLike,
    doses2: ArrayLike,
    parameters: U2OETMarginal,
    centering: str,
) -> FloatArray:
    powers = np.asarray(parameters.powers)
    x = u2oet_standardize(doses1, float(powers[0]))
    y = u2oet_standardize(doses2, float(powers[1]))
    x, y = (np.log(x), np.log(y)) if centering == "log" else (x - 1, y - 1)
    b = np.asarray(parameters.slopes)
    with np.errstate(over="ignore", invalid="ignore"):
        eta = (
            np.asarray(parameters.intercepts)
            + x[:, None, None] * b[:, 0]
            + y[None, :, None] * b[:, 1]
            + parameters.interaction * (x[:, None, None] * y[None, :, None])
        )
    if not np.all(np.isfinite(eta)):
        raise ArithmeticError("ordinal predictor exceeds floating-point range")
    return _ordinal(eta, parameters.link)


def _fgm_sides(log_mass: FloatArray) -> tuple[FloatArray, FloatArray]:
    below = np.logaddexp.accumulate(log_mass, axis=-1)
    above = np.logaddexp.accumulate(log_mass[..., ::-1], axis=-1)[..., ::-1]
    zero = np.full((*log_mass.shape[:-1], 1), -np.inf)
    # 1-a = F(y-1)+F(y); 1+a = S(y-1)+S(y), a=1-F(y-1)-F(y).
    minus = np.logaddexp(below, np.concatenate((zero, below[..., :-1]), -1))
    plus = np.logaddexp(above, np.concatenate((above[..., 1:], zero), -1))
    return minus, plus


def _joint(e: FloatArray, t: FloatArray, rho: float) -> FloatArray:
    independent = e[..., :, None] + t[..., None, :]
    if rho == 0:
        return independent
    em, ep = (a[..., :, None] for a in _fgm_sides(e))
    tm, tp = (a[..., None, :] for a in _fgm_sides(t))
    # Nonnegative decompositions of 1 +/- a*b preserve extreme tail cells,
    # including rho=+/-1, where direct subtraction would round to zero.
    extreme = np.logaddexp(em + tm, ep + tp) if rho > 0 else np.logaddexp(em + tp, ep + tm)
    extreme -= np.log(2)
    independent_weight = -np.inf if abs(rho) == 1 else np.log1p(-abs(rho))
    return independent + np.logaddexp(independent_weight, np.log(abs(rho)) + extreme)


@dataclass(frozen=True)
class U2OETProbabilities:
    """Axes are agent 1 dose, agent 2 dose, efficacy category, toxicity category.

    Categories and dose indices start at zero. These are conditional model
    probabilities at specified parameters, not posterior summaries.
    """

    log_efficacy: FloatArray
    log_toxicity: FloatArray
    log_joint: FloatArray

    @property
    def joint(self) -> FloatArray:
        return _freeze(np.exp(self.log_joint))

    def expected_utility(self, utility: ArrayLike) -> FloatArray:
        """Utility matrix rows are efficacy; columns are toxicity."""
        u = _real(utility, "utility")
        if u.shape != self.log_joint.shape[-2:]:
            raise ValueError("utility must have efficacy-by-toxicity shape")
        answer = np.sum(self.joint * u, axis=(-2, -1))
        if not np.all(np.isfinite(answer)):
            raise ArithmeticError("expected utility exceeds floating-point range")
        return _freeze(answer)

    def loglikelihood(self, counts: ArrayLike, *, toxicity_only: ArrayLike | None = None) -> float:
        """Grouped joint/optional toxicity-only data, without combinatorial constants."""
        n = _real(counts, "counts")
        if n.shape != self.log_joint.shape or np.any(n < 0) or np.any(n != np.floor(n)):
            raise ValueError("counts must be nonnegative integers with the joint probability shape")
        observed = n > 0
        value = float(np.sum(n[observed] * self.log_joint[observed]))
        if toxicity_only is not None:
            nt = _real(toxicity_only, "toxicity_only")
            if nt.shape != self.log_toxicity.shape or np.any(nt < 0) or np.any(nt != np.floor(nt)):
                raise ValueError("toxicity_only must be integer dose-by-dose-by-toxicity counts")
            observed_t = nt > 0
            value += float(np.sum(nt[observed_t] * self.log_toxicity[observed_t]))
        return value


def u2oet_probabilities(
    doses1: ArrayLike,
    doses2: ArrayLike,
    *,
    efficacy: U2OETMarginal,
    toxicity: U2OETMarginal,
    association: float = 0.0,
    model: Literal["pds", "cmi", "pds+cmi"] = "pds",
    centering: Literal["log", "linear"] = "log",
) -> U2OETProbabilities:
    """PDS, CMI or hybrid GCR marginals coupled with the paper's FGM copula.

    CMI requires unit powers and logarithmic centering. PDS requires zero
    interaction. The GAO/Gaussian-copula model is a separate, unsupported model.
    """
    rho = _real(association, "association")
    if rho.ndim or abs(rho) > 1:
        raise ValueError("association must be a scalar in [-1,1]")
    if model not in ("pds", "cmi", "pds+cmi") or centering not in ("log", "linear"):
        raise ValueError("unknown model or centering")
    for parameters in (efficacy, toxicity):
        if not isinstance(parameters, U2OETMarginal):
            raise ValueError("efficacy and toxicity must be U2OETMarginal parameters")
        if model == "pds" and parameters.interaction != 0:
            raise ValueError("PDS has no explicit interaction; use pds+cmi")
        if model == "cmi" and (np.any(np.asarray(parameters.powers) != 1) or centering != "log"):
            raise ValueError("CMI requires unit powers and logarithmic centering")
    e = _marginal(doses1, doses2, efficacy, centering)
    t = _marginal(doses1, doses2, toxicity, centering)
    return U2OETProbabilities(_freeze(e), _freeze(t), _freeze(_joint(e, t, float(rho))))
