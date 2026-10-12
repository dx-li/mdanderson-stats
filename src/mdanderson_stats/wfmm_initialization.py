"""Compose MOM, shrinkage, conditional covariance fitting and WFMM priors."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import cho_factor, cho_solve
from scipy.special import expit

from ._validation import FloatArray, scalar
from .wfmm_conditional import WFMMConditionalVariance, fit_wfmm_conditional_variances
from .wfmm_empirical_bayes import WFMMShrinkage, _coefficient_labels, calibrate_wfmm_shrinkage
from .wfmm_model import WFMMPrior, _finite_real, _strata
from .wfmm_mom import WFMMMomentInitialization, _basis, wfmm_mom_initialization
from .wfmm_native_eb import WFMMNativeEmpiricalBayes, wfmm_native_empirical_bayes
from .wfmm_variance_init import _frozen
from .wfmm_variance_prior import wfmm_variance_prior


@dataclass(frozen=True)
class WFMMInitialization:
    """Audited initialization; readiness is numerical, not clinical validation.

    MCMC still requires independent mixing/precision checks. A native signed
    stop or frozen zero proposal prevents strict ``ready_for_sampling``. The
    candidate estimates and explicit compatibility diagnostics remain available.
    """

    moment: WFMMMomentInitialization
    initial_shrinkage: WFMMShrinkage | WFMMNativeEmpiricalBayes
    conditional: WFMMConditionalVariance
    final_shrinkage: WFMMShrinkage | WFMMNativeEmpiricalBayes
    prior: WFMMPrior
    fixed_effect_estimates: FloatArray
    conditional_sampling_variances: FloatArray
    initial_shrunken_coefficients: FloatArray
    calibration_converged: bool
    ready_for_sampling: bool
    holds: tuple[str, ...]
    profile: str


def _native_gls(y, x, z, rg, sg, q, s, threshold):
    qx, rx = _basis(x, "fixed_design")
    estimates, sampling = np.empty((x.shape[1], y.shape[1])), np.empty((x.shape[1], y.shape[1]))
    for j in range(y.shape[1]):
        covariance = np.diag(s[sg, j])
        for group in range(q.shape[0]):
            if q[group, j] > threshold:
                block = z[:, rg == group]
                covariance += q[group, j] * (block @ block.T)
        if np.linalg.cond(covariance) > 1e12:
            raise ArithmeticError("native GLS covariance is ill conditioned")
        factor = cho_factor(covariance, lower=True, check_finite=False)
        information = qx.T @ cho_solve(factor, qx, check_finite=False)
        if np.linalg.cond(information) > 1e12:
            raise ArithmeticError("native GLS information is ill conditioned")
        b = np.linalg.solve(information, qx.T @ cho_solve(factor, y[:, j], check_finite=False))
        estimates[:, j] = np.linalg.solve(rx, b)
        diagonal = np.diag(rx.T @ information @ rx)
        if np.any(diagonal <= 0) or not np.all(np.isfinite(diagonal)):
            raise ArithmeticError("native GLS conditional information is not representable")
        sampling[:, j] = 1 / diagonal
    if (
        not np.all(np.isfinite(estimates))
        or not np.all(np.isfinite(sampling))
        or np.any(sampling <= 0)
    ):
        raise ArithmeticError("native GLS estimates are not representable")
    return estimates, sampling


def initialize_wfmm(
    coefficients: ArrayLike,
    fixed_design: ArrayLike,
    random_design: ArrayLike | None = None,
    *,
    random_strata: ArrayLike | None = None,
    residual_strata: ArrayLike | None = None,
    coefficient_partition: ArrayLike | None = None,
    profile: str = "stable",
    min_variance: float = 1e-6,
    zero_threshold: float = 1e-6,
    moment_tolerance: float = 1e-3,
    moment_iterations: int = 100,
    optimizer_iterations: int = 200,
    empirical_bayes_iterations: int = 1000,
    proposal_multiplier: float = 1.5,
    delta_omega: float = 1e-4,
    native_eb_settings: Mapping[str, float | int] | None = None,
) -> WFMMInitialization:
    """Create starting variances, calibrated priors and audited proposal scales.

    Stable mode composes complete MOM equations and mathematical EM with a
    fixed-shrunken-beta conditional covariance optimizer. Legacy mode reproduces
    recovered MOM/EB policies and native floor/bound restrictions, with an
    independent analytic-gradient optimizer rather than native Brent rounding.
    ``native_eb_settings`` is accepted only for legacy_3_1 and can set the native
    floors, odds cap and unsmoothed partitions. Inputs are transformed/selected
    coefficients; transform metadata and saved-study orchestration are separate.
    """
    if profile not in ("stable", "legacy_3_1"):
        raise ValueError("profile must be stable or legacy_3_1")
    floor = scalar(min_variance, "min_variance")
    if isinstance(min_variance, (bool, np.bool_)) or not 1e-100 <= floor <= 1e100:
        raise ValueError("min_variance must be a finite bounded positive floor")
    settings: dict[str, Any] = dict(native_eb_settings or {})
    allowed = {
        "initial_inclusion_probability",
        "initial_slab_to_sampling_variance",
        "min_inclusion",
        "min_ratio",
        "vague_ratio",
        "max_odds",
        "unsmoothed_partitions",
    }
    if set(settings) - allowed or (settings and profile != "legacy_3_1"):
        raise ValueError("native_eb_settings requires legacy_3_1 and recognized native settings")
    moment = wfmm_mom_initialization(
        coefficients,
        fixed_design,
        random_design,
        random_strata=random_strata,
        residual_strata=residual_strata,
        profile=profile,
        min_variance=min_variance,
        zero_threshold=zero_threshold,
        tolerance=moment_tolerance,
        max_iterations=moment_iterations,
    )
    y, x = _finite_real(coefficients, "coefficients"), _finite_real(fixed_design, "fixed_design")
    z = (
        np.empty((len(y), 0))
        if random_design is None
        else _finite_real(random_design, "random_design")
    )
    rg = _strata(random_strata, z.shape[1], "random_strata")
    sg = _strata(residual_strata, len(y), "residual_strata")
    labels = _coefficient_labels(coefficient_partition, y.shape[1])
    partitions, indices = np.unique(labels, return_inverse=True)

    def calibrate(q, s):
        if profile == "legacy_3_1":
            estimates, sampling = _native_gls(y, x, z, rg, sg, q, s, zero_threshold)
            native = wfmm_native_empirical_bayes(
                estimates / np.sqrt(sampling),
                profile=profile,
                coefficient_partition=labels,
                max_iterations=empirical_bayes_iterations,
                **settings,
            )
            pi = native.group_inclusion_probability[:, indices]
            ratio = native.group_slab_to_sampling_variance[:, indices]
            return (
                native,
                estimates,
                sampling,
                pi,
                ratio,
                native.inclusion_probability,
                native.converged,
            )
        mathematical = calibrate_wfmm_shrinkage(
            y,
            x,
            z if z.size else None,
            random_variance=q if q.size else None,
            residual_variance=s,
            random_strata=rg if rg.size else None,
            residual_strata=sg,
            coefficient_partition=labels,
            max_iterations=empirical_bayes_iterations,
        )
        pi = mathematical.inclusion_probability
        ratio = mathematical.slab_to_sampling_variance
        gamma = np.zeros_like(pi)
        gamma[pi == 1] = 1
        interior = (pi > 0) & (pi < 1)
        gamma[interior] = expit(
            np.log(pi[interior])
            - np.log1p(-pi[interior])
            - 0.5 * np.log1p(ratio[interior])
            + 0.5
            * mathematical.standardized_estimates[interior] ** 2
            * (ratio[interior] / (1 + ratio[interior]))
        )
        return (
            mathematical,
            mathematical.fixed_estimates,
            mathematical.conditional_variances,
            pi,
            ratio,
            gamma,
            bool(mathematical.group_converged.all()),
        )

    first, beta, _, _, ratio, gamma, first_ok = calibrate(
        moment.random_variance, moment.residual_variance
    )
    shrunken = beta * gamma * (ratio / (1 + ratio))
    conditional = fit_wfmm_conditional_variances(
        y,
        x,
        shrunken,
        z if z.size else None,
        random_variance=moment.random_variance if moment.random_variance.size else None,
        residual_variance=moment.residual_variance,
        random_strata=rg if rg.size else None,
        residual_strata=sg,
        profile=profile,
        min_variance=min_variance,
        proposal_multiplier=proposal_multiplier,
        max_iterations=optimizer_iterations,
    )
    final, beta, sampling, pi, ratio, _, final_ok = calibrate(
        conditional.random_variance, conditional.residual_variance
    )
    prior = wfmm_variance_prior(
        inclusion_probability=pi,
        slab_variance=sampling * ratio,
        residual_variance=conditional.residual_variance,
        residual_stratum_sizes=np.bincount(sg),
        random_variance=conditional.random_variance if rg.size else None,
        random_effect_counts=np.bincount(rg) if rg.size else None,
        delta_omega=delta_omega,
    )
    holds = []
    if not moment.converged:
        holds.append("moment_not_converged")
    if not first_ok:
        holds.append("initial_empirical_bayes_not_converged")
    if not conditional.optimizer_success.all():
        holds.append("conditional_optimizer_not_converged")
    if not conditional.positive_proposals:
        holds.append("proposal_scales_not_positive_and_usable")
    if conditional.frozen_descent.any():
        holds.append("native_floor_freeze_blocks_likelihood_improvement")
    if not final_ok:
        holds.append("final_empirical_bayes_not_converged")
    return WFMMInitialization(
        moment,
        first,
        conditional,
        final,
        prior,
        _frozen(beta),
        _frozen(sampling),
        _frozen(shrunken),
        first_ok and final_ok,
        not holds,
        tuple(holds),
        profile,
    )
