"""Explicit pseudo-trial prior-mean calibration for the current UAROET model."""

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from ._cdflib import _freeze
from ._validation import FloatArray
from .uaroet import _integer, _real_matrix
from .uaroet_desktop_study import UAROETDesktopScenario, UAROETDesktopStudy
from .uaroet_precision import UAROETPrecisionFit, fit_uaroet_desktop


def _pseudo_start(counts: np.ndarray) -> FloatArray:
    """Interior continuation-logit initialization; never used as a fitted prior."""
    result = []
    for margin in (counts.sum(2), counts.sum(1)):
        a = margin + 0.5
        tail = np.flip(np.cumsum(np.flip(a, axis=1), axis=1), axis=1)
        logits = np.log(tail[:, 1:] / a[:, :-1])
        monotone = np.maximum.accumulate(logits, axis=0)
        result.extend(monotone[0])
        result.extend(np.maximum(np.diff(monotone, axis=0), 0.1).ravel())
    return np.asarray(result)


@dataclass(frozen=True)
class UAROETCalibration:
    """Pseudo-posterior means and explicit replication/precision evidence."""

    replicate_parameters: FloatArray
    replicate_association: FloatArray
    maximum_mcse_ratios: FloatArray
    maximum_utility_rhat: FloatArray
    draws_per_chain: tuple[int, ...]
    scenario_identifier: int
    dose_count: int
    efficacy_levels: int
    toxicity_levels: int

    @property
    def prior_mean(self) -> FloatArray:
        return _freeze(self.replicate_parameters.mean(0))

    @property
    def meets_guide_replication_minimum(self) -> bool:
        return len(self.replicate_parameters) >= 1000

    def apply(self, study: UAROETDesktopStudy) -> UAROETDesktopStudy:
        """Return a new study with calibrated means, preserving supplied SDs."""
        if (study.dose_count, study.efficacy_levels, study.toxicity_levels) != (
            self.dose_count,
            self.efficacy_levels,
            self.toxicity_levels,
        ):
            raise ValueError("calibration and study dimensions differ")
        return replace(
            study,
            prior_mean=self.prior_mean,
            native_association_mean=float(self.replicate_association.mean()),
        )


def calibrate_uaroet_prior(
    study: UAROETDesktopStudy,
    scenario: UAROETDesktopScenario,
    *,
    repetitions: int,
    rng: np.random.Generator,
    target: float = 0.03,
    chunk_draws: int = 1000,
    max_draws: int = 100_000,
    max_evaluations_per_trial: int = 8_000_000,
    maximum_rhat: float = 1.05,
    on_trial: Callable[[int, UAROETPrecisionFit], None] | None = None,
) -> UAROETCalibration:
    """Average precision-qualified pseudo-posteriors from100 patients per dose.

    The pseudo-prior has mean0/SD100 and Uniform(-1,1) association. Every trial
    uses independent generated outcomes from the supplied Gaussian-copula
    scenario. This driver requires at least two chains and finite utility
    R-hat within the explicitly supplied bound, in addition to per-chain
    utility precision. These are Python quality gates, not native equivalence.
    Fewer than1000 repetitions are exploratory under the guide. A failed trial
    raises and never returns a partially calibrated prior. The optional callback
    receives each completed fit for progress/checkpointing by the caller.
    """
    if (
        not isinstance(study, UAROETDesktopStudy)
        or not isinstance(scenario, UAROETDesktopScenario)
        or not isinstance(rng, np.random.Generator)
    ):
        raise ValueError("require a study, scenario and explicit NumPy Generator")
    reps = _integer(repetitions, "repetitions", 1, 100_000)
    if scenario.efficacy.shape != (
        study.dose_count,
        study.efficacy_levels,
    ) or scenario.toxicity.shape != (study.dose_count, study.toxicity_levels):
        raise ValueError("scenario must match study dimensions")
    if int(study.simulation_parameters.get("NumChains", 4)) < 2:
        raise ValueError("calibration quality gate requires at least two chains")
    rhat = _real_matrix(maximum_rhat, "maximum_rhat")
    if rhat.ndim or float(rhat) < 1:
        raise ValueError("maximum_rhat must be finite and at least one")
    maximum_rhat = float(rhat)
    ratio = _real_matrix(target, "target")
    if ratio.ndim or not 0 < float(ratio) < 1:
        raise ValueError("target must be strictly between zero and one")
    target = float(ratio)
    chunk_draws = _integer(chunk_draws, "chunk_draws", 8, 100_000)
    max_draws = _integer(max_draws, "max_draws", 8, 1_000_000)
    max_evaluations_per_trial = _integer(
        max_evaluations_per_trial, "max_evaluations_per_trial", 1, 128_000_000
    )
    if max_draws < max(chunk_draws, int(study.simulation_parameters.get("BurnInSamples", 1000))):
        raise ValueError("max_draws must cover burn-in and chunk length")
    if (
        max_draws * study.dose_count * int(study.simulation_parameters.get("NumChains", 4))
        > 20_000_000
    ):
        raise ValueError("retained utility budget exceeds twenty million values")
    if chunk_draws * study.dose_count * study.efficacy_levels * study.toxicity_levels > 2_000_000:
        raise ValueError("joint chunk exceeds two million cells")
    if on_trial is not None and not callable(on_trial):
        raise ValueError("on_trial must be callable")
    joint = scenario.joint()
    parameter_means = []
    associations = []
    ratios = []
    rhats = []
    draws = []
    for repetition in range(reps):
        counts = np.stack([rng.multinomial(100, p.ravel()).reshape(p.shape) for p in joint])
        patients: list[tuple[int, int, int, int]] = []
        for d, e, t in np.ndindex(counts.shape):
            for _ in range(int(counts[d, e, t])):
                patients.append((len(patients) + 1, d, e, t))
        pseudo = replace(
            study,
            prior_mean=np.zeros_like(study.prior_mean),
            prior_sd=np.full_like(study.prior_sd, 100),
            patients=tuple(patients),
            trial_parameters={**study.trial_parameters, "NumPatients": 100 * study.dose_count},
        )
        fit = fit_uaroet_desktop(
            pseudo,
            rng=rng,
            target=target,
            conduct=False,
            chunk_draws=chunk_draws,
            max_draws=max_draws,
            max_evaluations=max_evaluations_per_trial,
            initial_parameters=_pseudo_start(counts),
            initial_association=0,
        )
        if (
            not fit.precision_met
            or not np.all(np.isfinite(fit.utility_split_rhat))
            or np.max(fit.utility_split_rhat) > maximum_rhat
        ):
            raise RuntimeError(
                f"pseudo-trial {repetition + 1} failed precision or between-chain quality gate"
            )
        parameter_means.append(fit.mean_parameters)
        associations.append(fit.mean_association)
        ratios.append(float(np.max(fit.utility_mcse_ratio[:, fit.monitored_doses])))
        rhats.append(float(np.max(fit.utility_split_rhat)))
        draws.append(fit.draws_per_chain)
        if on_trial is not None:
            on_trial(repetition + 1, fit)
    return UAROETCalibration(
        _freeze(np.stack(parameter_means)),
        _freeze(np.asarray(associations)),
        _freeze(np.asarray(ratios)),
        _freeze(np.asarray(rhats)),
        tuple(draws),
        scenario.identifier,
        study.dose_count,
        study.efficacy_levels,
        study.toxicity_levels,
    )


def uaroet_calibrated_prior_text(result: UAROETCalibration) -> str:
    """Native threshold-major coordinates plus stored association mean,17-digit text."""
    if not isinstance(result, UAROETCalibration):
        raise ValueError("result must be UAROETCalibration")
    means = result.prior_mean
    values: list[float] = []
    offset = 0
    for levels in (result.efficacy_levels, result.toxicity_levels):
        width = result.dose_count * (levels - 1)
        values.extend(
            means[offset : offset + width].reshape(result.dose_count, levels - 1).T.ravel()
        )
        offset += width
    values.append(float(result.replicate_association.mean()))
    return "".join(f"{value:.17g}\n" for value in values)
