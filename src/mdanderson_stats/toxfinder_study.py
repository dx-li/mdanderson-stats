"""Saved ToxFinder physician elicitation and explicit Python simulation settings."""

from dataclasses import dataclass
from html import escape
from typing import Literal

import numpy as np

from ._validation import finite
from .toxfinder_model import ToxFinderPrior
from .toxfinder_native_elicitation import ToxFinderNativeElicitation, elicit_toxfinder_native_prior
from .toxfinder_native_io import (
    ToxFinderNativeStudy,
    ToxFinderObservedRecommendation,
    ToxFinderPatientRecord,
    read_toxfinder_tmml,
)
from .toxfinder_native_trial import ToxFinderNativeSimulation, simulate_toxfinder_native


def _study(study: ToxFinderNativeStudy) -> ToxFinderNativeStudy:
    if not isinstance(study, ToxFinderNativeStudy):
        raise TypeError("study must be ToxFinderNativeStudy")
    return read_toxfinder_tmml(study.to_tmml())


def elicit_toxfinder_study_prior(
    study: ToxFinderNativeStudy, *, seed: int, max_probability_evaluations: int = 8000
) -> ToxFinderNativeElicitation:
    """Calibrate all saved physician inputs, interpreting stored variances literally.

    The original GUI displays square roots for pr-alpha3-var/pr-beta3-var but
    writes variances to TMML. Native seeds/MCMC controls remain metadata; this
    seed explicitly selects independent NumPy Gamma streams.
    """
    study = _study(study)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be an integer in [0,4294967295]")
    settings = dict(study.settings)

    def values(key: str, size: int) -> np.ndarray:
        if key not in settings or len(settings[key]) != size:
            raise ValueError(f"TMML requires {key} with {size} values for prior calibration")
        return finite([float(x) for x in settings[key]], key)

    doses = np.array(
        [
            values("pr-negtox", 2),
            study.reference_doses,
            values("pr-hightox", 2),
            values("pr-toxic", 2),
        ]
    )
    means = [float(values(key, 1)[0]) for key in ("pr-alpha3", "pr-beta3")]
    variances = [float(values(key, 1)[0]) for key in ("pr-alpha3-var", "pr-beta3-var")]
    low = float(values("toxic-minprob", 1)[0])
    high = float(values("toxic-maxprob", 1)[0])
    if not 0 < low < study.target < high < 1:
        raise ValueError("require negligible probability < target < high toxicity probability")
    return elicit_toxfinder_native_prior(
        doses[:, 0],
        doses[:, 1],
        study.target,
        interaction_mean=means,
        interaction_variance=variances,
        negligible_probability=low,
        high_toxicity_probability=high,
        rng=np.random.default_rng(seed),
        max_probability_evaluations=max_probability_evaluations,
    )


@dataclass(frozen=True)
class ToxFinderStudyRun:
    study: ToxFinderNativeStudy
    prior: ToxFinderPrior
    simulation: ToxFinderNativeSimulation
    seed: int
    draws: int
    warmup: int
    chains: int
    elicitation: ToxFinderNativeElicitation | None = None

    def to_html(self) -> str:
        """Report simulation, captured inputs and quality diagnostics together."""
        report = self.simulation.to_html(reference_doses=self.study.reference_doses)
        settings = (
            f"<h2>Captured Python inference settings</h2><p>Seed {self.seed}; "
            f"draws {self.draws}; warmup {self.warmup}; chains {self.chains}. "
            "Original MCMC controls and random seeds are retained below as metadata.</p>"
            "<h2>Prior moments</h2><p>Order: alpha1, beta1, alpha2, beta2, alpha3, beta3.</p>"
            f"<p>Mean: {escape(str(self.prior.mean.tolist()))}; "
            f"variance: {escape(str(self.prior.variance.tolist()))}.</p>"
            "<h2>Final posterior diagnostics per trial</h2><table><tr>"
            "<th>Trial</th><th>Split R-hat</th><th>Parameter MCSE</th></tr>"
        )
        for i, trial in enumerate(self.simulation.trials):
            summary = trial.final_fit.parameter_summary
            settings += (
                f"<tr><td>{i}</td><td>{escape(str(summary.split_rhat))}</td>"
                f"<td>{escape(str(summary.batch_mean_mcse))}</td></tr>"
            )
        settings += "</table>"
        if self.elicitation is not None:
            settings += "<h2>Independent prior-calibration assessment</h2>"
            for i, agent in enumerate((self.elicitation.agent1, self.elicitation.agent2)):
                settings += (
                    f"<p>Agent {i + 1}: tail probabilities "
                    f"{escape(str(agent.assessed_tail_probability))}; "
                    f"MCSE {escape(str(agent.assessed_probability_mcse))}; "
                    f"target residual {escape(str(agent.assessed_target_residual))}; "
                    f"estimated cutoff met: {agent.assessed_cutoff_met}.</p>"
                )
        return (
            report + settings + "<h2>Saved TMML</h2><pre>" + escape(self.study.to_tmml()) + "</pre>"
        )


def simulate_toxfinder_study(
    study: ToxFinderNativeStudy,
    *,
    prior: ToxFinderPrior | ToxFinderNativeElicitation,
    trials: int,
    seed: int,
    draws: int = 500,
    warmup: int = 250,
    chains: int = 2,
    final_policy: Literal["refit", "legacy"] = "refit",
    max_sampler_evaluations: int = 20_000_000,
) -> ToxFinderStudyRun:
    """Execute saved design/scenario with an explicit prior and Python sampler.

    Refit after all outcomes by default. Legacy pre-final-cohort posterior use
    requires an explicit final_policy='legacy'; the report identifies the choice.
    """
    study = _study(study)
    elicitation = prior if isinstance(prior, ToxFinderNativeElicitation) else None
    actual_prior = elicitation.prior if elicitation is not None else prior
    if not isinstance(actual_prior, ToxFinderPrior):
        raise TypeError("prior must be ToxFinderPrior or ToxFinderNativeElicitation")
    result = simulate_toxfinder_native(
        study.dose_levels,
        study.true_surface,
        prior=actual_prior,
        target=study.target,
        stage1_patients=study.stage1_patients,
        stage2_patients=study.stage2_patients,
        cohort_size=study.cohort_size,
        trials=trials,
        seed=seed,
        start_index=study.start_index,
        potency=study.potency,
        mode=study.mode,
        final_policy=final_policy,
        draws=draws,
        warmup=warmup,
        chains=chains,
        max_sampler_evaluations=max_sampler_evaluations,
    )
    return ToxFinderStudyRun(study, actual_prior, result, seed, draws, warmup, chains, elicitation)


def write_toxfinder_subject_csv(
    records: tuple[ToxFinderPatientRecord, ...], *, reference_doses
) -> str:
    """Write editable native patient rows and validate through the strict reader."""
    import csv
    import io

    from .toxfinder_native_io import ToxFinderPatientRecord, _scale, read_toxfinder_subject_csv

    scale = _scale(reference_doses)
    if len(records) > 100:
        raise ValueError("at most 100 patient records")
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    for r in records:
        if not isinstance(r, ToxFinderPatientRecord):
            raise TypeError("records must contain ToxFinderPatientRecord")
        dose = finite(r.standardized_dose, "standardized_dose")
        if dose.shape != (2,) or np.any(dose < 0) or np.any(dose > 1):
            raise ValueError("standardized doses must be a pair in [0,1]")
        physical = dose * scale
        writer.writerow(
            [
                r.patient_id,
                r.identifier,
                r.year,
                r.month,
                r.day,
                r.time,
                r.cohort_id,
                repr(float(physical[0])),
                repr(float(physical[1])),
                r.toxicity,
            ]
        )
    text = output.getvalue()
    restored = read_toxfinder_subject_csv(text, reference_doses=scale)
    if len(restored) != len(records):
        raise ValueError("patient IDs must not be interpreted as CSV comments")
    return text


@dataclass(frozen=True)
class ToxFinderStudyRecommendation:
    study: ToxFinderNativeStudy
    recommendation: ToxFinderObservedRecommendation
    prior: ToxFinderPrior
    seed: int
    draws: int
    warmup: int
    chains: int

    def to_html(self) -> str:
        """Printable current-history recommendation, physical doses and diagnostics."""
        r = self.recommendation
        dose = None if r.dose is None else (r.dose * self.study.reference_doses).tolist()
        body = (
            "<!doctype html><meta charset='utf-8'><title>ToxFinder patient recommendation</title>"
            "<h1>ToxFinder patient recommendation</h1>"
            f"<p>Reason: {escape(r.reason)}; physical dose pair: {escape(str(dose))}; "
            f"observed: {r.observed_patients}; pending: {r.pending_patients}.</p>"
            f"<p>Python seed {self.seed}; draws {self.draws}; warmup {self.warmup}; "
            f"chains {self.chains}. Pending outcomes are excluded from inference.</p>"
            f"<p>Parameter split R-hat: {escape(str(r.fit.parameter_summary.split_rhat))}; "
            f"MCSE: {escape(str(r.fit.parameter_summary.batch_mean_mcse))}.</p>"
            f"<p>Prior mean: {escape(str(self.prior.mean))}; "
            f"variance: {escape(str(self.prior.variance))}.</p>"
            "<h2>Captured patient CSV (physical doses)</h2><pre>"
        )
        return (
            body
            + escape(
                write_toxfinder_subject_csv(r.records, reference_doses=self.study.reference_doses)
            )
            + "</pre><h2>Saved TMML</h2><pre>"
            + escape(self.study.to_tmml())
            + "</pre>"
        )


def recommend_toxfinder_study(
    study: ToxFinderNativeStudy,
    records: tuple[ToxFinderPatientRecord, ...],
    *,
    prior: ToxFinderPrior | ToxFinderNativeElicitation,
    seed: int,
    draws: int = 500,
    warmup: int = 250,
    chains: int = 2,
) -> ToxFinderStudyRecommendation:
    """Recommend from observed saved patients under the saved design parameters."""
    from .toxfinder_native_io import recommend_toxfinder_native

    study = _study(study)
    actual_prior = prior.prior if isinstance(prior, ToxFinderNativeElicitation) else prior
    if not isinstance(actual_prior, ToxFinderPrior):
        raise TypeError("prior must be ToxFinderPrior or ToxFinderNativeElicitation")
    r = recommend_toxfinder_native(
        records,
        study.dose_levels,
        prior=actual_prior,
        target=study.target,
        stage1_patients=study.stage1_patients,
        stage2_patients=study.stage2_patients,
        cohort_size=study.cohort_size,
        seed=seed,
        start_index=study.start_index,
        potency=study.potency,
        mode=study.mode,
        draws=draws,
        warmup=warmup,
        chains=chains,
    )
    return ToxFinderStudyRecommendation(study, r, actual_prior, seed, draws, warmup, chains)
