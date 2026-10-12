"""Bounded two-stage ToxFinder simulations using independent Python inference."""

import html
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray, finite
from .toxfinder_model import ToxFinderFit, ToxFinderPrior, fit_toxfinder, toxfinder_probabilities
from .toxfinder_native_decision import (
    _goal,
    _levels,
    toxfinder_native_allocation,
    toxfinder_native_stage1,
)
from .toxfinder_native_information import ToxFinderNativeMAP, _frozen


def _integer(value: int, name: str, low: int, high: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= value <= high:
        raise ValueError(f"{name} must be in [{low},{high}]")
    return int(value)


def _minimum_sampler_work(
    prior: ToxFinderPrior,
    n1: int,
    n2: int,
    cohort: int,
    draws: int,
    warmup: int,
    chains: int,
    final_policy: str,
) -> int:
    """Exact schedule lower bound, including one chain initialization per fit."""
    if n1 % cohort or n2 % cohort or n1 + n2 > 100:
        raise ValueError("stage sizes must be cohort multiples, with at most 100 patients")
    if final_policy not in ("legacy", "refit"):
        raise ValueError("unsupported final posterior policy")
    free = int(np.count_nonzero(prior.variance))
    rows = list(range(0, n1, cohort)) + list(range(n1, n1 + n2, 2 * cohort))
    if final_policy == "refit":
        rows.append(n1 + n2)
    return sum(chains if n == 0 else chains * (1 + 3 * free * (draws + warmup)) * n for n in rows)


def _check_sampler_budget(minimum: int, budget: int) -> None:
    if minimum > budget:
        raise ValueError(
            f"sampler budget {budget} is below the deterministic minimum {minimum} "
            "objective-evaluation rows for this design. Increase max_sampler_evaluations "
            "explicitly (maximum 200000000), allowing additional work for slice stepping "
            "and rejected proposals; the minimum is not a sufficient execution budget."
        )


def toxfinder_native_surface(surface: ArrayLike, doses: ArrayLike) -> FloatArray:
    """Bilinear probabilities on uniform [0,1] axes; surface rows index drug 2."""
    raw = np.asarray(surface)
    if raw.ndim != 2 or not all(2 <= size <= 100 for size in raw.shape):
        raise ValueError("surface must have 2..100 rows and columns")
    values = finite(raw, "surface")
    d = finite(doses, "doses")
    if d.ndim != 2 or d.shape[1] != 2 or len(d) > 10_000:
        raise ValueError("doses must have shape (0..10,000,2)")
    if np.any(values < 0) or np.any(values > 1) or np.any(d < 0) or np.any(d > 1):
        raise ValueError("surface probabilities and standardized doses must be in [0,1]")
    x, y = d[:, 0] * (values.shape[1] - 1), d[:, 1] * (values.shape[0] - 1)
    ix = np.minimum(x.astype(int), values.shape[1] - 2)
    iy = np.minimum(y.astype(int), values.shape[0] - 2)
    fx, fy = x - ix, y - iy
    return _frozen(
        (1 - fy) * ((1 - fx) * values[iy, ix] + fx * values[iy, ix + 1])
        + fy * ((1 - fx) * values[iy + 1, ix] + fx * values[iy + 1, ix + 1])
    )


@dataclass(frozen=True)
class ToxFinderNativeCohort:
    first_patient: int
    selected_index: int
    recommended_doses: FloatArray
    dose: FloatArray
    toxicities: int
    subjects: int
    true_toxicity: float
    posterior_update: int
    map_fit: ToxFinderNativeMAP | None = None


@dataclass(frozen=True)
class ToxFinderNativeTrial:
    cohorts: tuple[ToxFinderNativeCohort, ...]
    final_doses: FloatArray
    estimated_toxicity: FloatArray
    posterior_sd_toxicity: FloatArray
    true_final_toxicity: FloatArray
    final_fit: ToxFinderFit
    posterior_subjects: int
    subjects: int
    toxicities: int
    mean_true_exposure: float
    final_policy: str
    sampler_evaluations: int

    def to_html(
        self, *, reference_doses: ArrayLike = (1.0, 1.0), title: str = "ToxFinder trial"
    ) -> str:
        """Self-contained escaped report; dose scaling is explicit."""
        scale = finite(reference_doses, "reference_doses")
        if scale.shape != (2,) or np.any(scale <= 0):
            raise ValueError("reference_doses must contain two positive scales")
        rows = []
        for label, dose, p, sd, truth in zip(
            ("Middle", "Branch 1", "Branch 2"),
            self.final_doses * scale,
            self.estimated_toxicity,
            self.posterior_sd_toxicity,
            self.true_final_toxicity,
        ):
            rows.append(
                f"<tr><td>{label}</td><td>{dose[0]:.6g}</td><td>{dose[1]:.6g}</td>"
                f"<td>{p:.6g}</td><td>{sd:.6g}</td><td>{truth:.6g}</td>"
                f"<td>{abs(p - truth):.6g}</td></tr>"
            )
        return (
            "<!doctype html><meta charset='utf-8'><title>"
            + html.escape(title)
            + "</title><h1>"
            + html.escape(title)
            + "</h1>"
            + f"<p>{self.toxicities} toxicities / {self.subjects} patients; observed risk "
            f"{self.toxicities / self.subjects:.6g}; "
            f"mean true exposure {self.mean_true_exposure:.6g}. "
            f"Final posterior includes {self.posterior_subjects} patients "
            f"({html.escape(self.final_policy)}).</p>"
            + "<p>Independent Python slice sampling; native MCMC/RNG identity is not claimed. "
            "Inspect chain diagnostics and increase draws before interpreting "
            "Monte Carlo results.</p>"
            "<table><tr><th>Contour point</th><th>Drug 1</th><th>Drug 2</th>"
            "<th>Posterior mean</th><th>Posterior SD</th><th>Scenario truth</th>"
            "<th>Absolute error</th></tr>"
            + "".join(rows)
            + "</table>"
            + "<h2>Allocation MAP diagnostics</h2>"
            + "".join(
                f"<p>First patient {c.first_patient} (zero-based): pass convergence "
                f"{c.map_fit.pass_converged}; iterations {c.map_fit.pass_iterations}; "
                f"objective evaluations {c.map_fit.evaluations}.</p>"
                for c in self.cohorts
                if c.map_fit is not None
            )
            + "<p>False denotes an unconverged native simplex pass; "
            "the compatibility allocation retains that result.</p>"
        )


def simulate_toxfinder_native_trial(
    dose_levels: ArrayLike,
    true_surface: ArrayLike,
    *,
    prior: ToxFinderPrior,
    target: float,
    stage1_patients: int,
    stage2_patients: int,
    cohort_size: int,
    seed: int,
    start_index: int = 0,
    potency: float = 1.0,
    mode: Literal["fi_ckp", "fi"] = "fi_ckp",
    final_policy: Literal["legacy", "refit"] = "legacy",
    draws: int = 500,
    warmup: int = 250,
    chains: int = 2,
    max_sampler_evaluations: int = 2_000_000,
) -> ToxFinderNativeTrial:
    """Run stage one by cohorts and stage two by pairs sharing one posterior.

    All doses are standardized. Stage counts must be cohort multiples. Legacy
    final summaries use the last preallocation posterior; refit is an explicit
    alternative. Outcomes and sampler use independent reproducible RNG streams.
    """
    levels, goal = _levels(dose_levels), _goal(target)
    n1 = _integer(stage1_patients, "stage1_patients", 1, 100)
    n2 = _integer(stage2_patients, "stage2_patients", 0, 100)
    cohort = _integer(cohort_size, "cohort_size", 1, 100)
    start = _integer(start_index, "start_index", 0, len(levels) - 1)
    seed = _integer(seed, "seed", 0, 2**63 - 1)
    budget = _integer(max_sampler_evaluations, "max_sampler_evaluations", 1, 200_000_000)
    _integer(draws, "draws", 8, 10_000)
    _integer(warmup, "warmup", 0, 100_000)
    _integer(chains, "chains", 2, 4)
    if n1 % cohort or n2 % cohort or n1 + n2 > 100:
        raise ValueError("stage sizes must be cohort multiples, with at most 100 patients")
    if mode not in ("fi", "fi_ckp") or final_policy not in ("legacy", "refit"):
        raise ValueError("unsupported allocation mode or final posterior policy")
    if not np.isfinite(potency) or potency <= 0:
        raise ValueError("potency must be positive and finite")
    # Validate all information-prior assumptions before any simulation work.
    from .toxfinder_native_information import _gamma

    _gamma(prior)
    _check_sampler_budget(
        _minimum_sampler_work(prior, n1, n2, cohort, draws, warmup, chains, final_policy), budget
    )
    if np.any(prior.mean < 0.005) or np.any(prior.mean > 30):
        raise ValueError("native MAP prior means must lie in [.005,30]")
    toxfinder_native_surface(true_surface, levels)
    streams = np.random.SeedSequence(seed).spawn(2)
    outcomes_rng, sampler_rng = (np.random.default_rng(s) for s in streams)
    history: list[list[float]] = []
    outcomes: list[int] = []
    cohorts: list[ToxFinderNativeCohort] = []
    evaluations = 0
    update = 0
    posterior_subjects = 0
    fit: ToxFinderFit

    def refit() -> ToxFinderFit:
        nonlocal evaluations, update, posterior_subjects
        d = np.asarray(history) if history else levels[start : start + 1]
        y = np.asarray(outcomes) if history else np.zeros(1)
        n = np.ones(len(history)) if history else np.zeros(1)
        # Deterministic lower bound rejects excessive requests before fitting.
        result = fit_toxfinder(
            d,
            y,
            n,
            prior=prior,
            rng=sampler_rng,
            draws=draws,
            warmup=warmup,
            chains=chains,
            max_evaluations=min(budget - evaluations, 20_000_000),
        )
        evaluations += result.evaluations * len(d)
        if evaluations > budget:
            raise ValueError("trial exceeds total sampler evaluation budget")
        update += 1
        posterior_subjects = len(history)
        return result

    def enroll(pairs: FloatArray, selected: int, map_fit: ToxFinderNativeMAP | None = None) -> None:
        dose = pairs[selected]
        truth = float(toxfinder_native_surface(true_surface, [dose])[0])
        y = (outcomes_rng.random(cohort) < truth).astype(int)
        cohorts.append(
            ToxFinderNativeCohort(
                len(history),
                selected,
                _frozen(pairs),
                _frozen(dose),
                int(y.sum()),
                cohort,
                truth,
                update,
                map_fit,
            )
        )
        history.extend([dose.tolist() for _ in range(cohort)])
        outcomes.extend(y.tolist())

    while len(history) < n1 + n2:
        fit = refit()
        if len(history) < n1:
            dose = toxfinder_native_stage1(
                levels,
                fit.log_parameters,
                history,
                outcomes,
                target=goal,
                start_index=start,
                log_parameters=True,
            )
            enroll(np.asarray([dose]), 0)
        else:
            result = toxfinder_native_allocation(
                levels,
                fit.log_parameters,
                target=goal,
                potency=potency,
                mode=mode,
                doses=history,
                toxicities=outcomes,
                subjects=np.ones(len(history)),
                prior=prior,
                log_parameters=True,
            )
            selected = 2 if ((len(history) - n1) // cohort) % 2 == 0 else 1
            enroll(result.doses, selected, result.map_fit)
            if len(history) < n1 + n2:
                enroll(result.doses, 3 - selected, result.map_fit)
    if final_policy == "refit":
        fit = refit()
    contour = toxfinder_native_allocation(
        levels,
        fit.log_parameters,
        target=goal,
        potency=potency,
        mode=mode,
        log_parameters=True,
        allocation=False,
    )
    probabilities = toxfinder_probabilities(contour.doses, fit.log_parameters, log_parameters=True)[
        ..., 1
    ].reshape((-1, 3))
    return ToxFinderNativeTrial(
        tuple(cohorts),
        contour.doses,
        contour.estimated_toxicity,
        _frozen(probabilities.std(axis=0, ddof=1)),
        toxfinder_native_surface(true_surface, contour.doses),
        fit,
        posterior_subjects,
        len(history),
        sum(outcomes),
        sum(c.true_toxicity * c.subjects for c in cohorts) / len(history),
        final_policy,
        evaluations,
    )


@dataclass(frozen=True)
class ToxFinderNativeSimulation:
    trials: tuple[ToxFinderNativeTrial, ...]
    mean_toxicities: float
    sd_toxicities: float
    mcse_mean_toxicities: float
    mean_true_exposure: float
    sd_true_exposure: float
    mean_final_doses: FloatArray
    sd_final_doses: FloatArray
    mean_estimated_toxicity: FloatArray
    mean_true_final_toxicity: FloatArray
    mean_absolute_error: FloatArray
    mean_posterior_sd: FloatArray

    def to_html(
        self, *, reference_doses: ArrayLike = (1.0, 1.0), title: str = "ToxFinder simulation"
    ) -> str:
        scale = finite(reference_doses, "reference_doses")
        if scale.shape != (2,) or np.any(scale <= 0):
            raise ValueError("reference_doses must contain two positive scales")
        rows = []
        diagnostics = [
            "<h2>Allocation MAP diagnostics</h2><table><tr><th>Trial</th>"
            "<th>First patient (zero-based)</th><th>Pass convergence</th>"
            "<th>Pass iterations</th><th>Objective evaluations</th></tr>"
        ]
        for trial_index, trial in enumerate(self.trials, 1):
            for cohort in trial.cohorts:
                fit = cohort.map_fit
                if fit is not None:
                    diagnostics.append(
                        f"<tr><td>{trial_index}</td><td>{cohort.first_patient}</td>"
                        f"<td>{fit.pass_converged}</td><td>{fit.pass_iterations}</td>"
                        f"<td>{fit.evaluations}</td></tr>"
                    )
        diagnostics.append(
            "</table><p>False denotes an unconverged native simplex pass; "
            "the compatibility allocation retains that result.</p>"
        )
        for i, label in enumerate(("Middle", "Branch 1", "Branch 2")):
            dose, sd = self.mean_final_doses[i] * scale, self.sd_final_doses[i] * scale
            rows.append(
                f"<tr><td>{label}</td><td>{dose[0]:.6g} ({sd[0]:.6g})</td>"
                f"<td>{dose[1]:.6g} ({sd[1]:.6g})</td>"
                f"<td>{self.mean_estimated_toxicity[i]:.6g}</td>"
                f"<td>{self.mean_true_final_toxicity[i]:.6g}</td>"
                f"<td>{self.mean_absolute_error[i]:.6g}</td>"
                f"<td>{self.mean_posterior_sd[i]:.6g}</td></tr>"
            )
        return (
            "<!doctype html><meta charset='utf-8'><title>"
            + html.escape(title)
            + "</title><h1>"
            + html.escape(title)
            + f"</h1><p>{len(self.trials)} trials. Mean toxicities "
            f"{self.mean_toxicities:.6g}; sample SD {self.sd_toxicities:.6g}; "
            f"MCSE of mean {self.mcse_mean_toxicities:.6g}. Mean true exposure "
            f"{self.mean_true_exposure:.6g}; sample SD {self.sd_true_exposure:.6g}.</p>"
            f"<p>Final posterior policy: {html.escape(self.trials[0].final_policy)}. "
            "SD uses n-1; native GUI denominator and RNG identity are not claimed. "
            "Per-trial fits retain parameter MCSE and split R-hat diagnostics.</p>"
            "<table><tr><th>Point</th><th>Mean drug 1 (SD)</th><th>Mean drug 2 (SD)</th>"
            "<th>Mean estimated risk</th><th>Mean true risk</th><th>Mean absolute error</th>"
            "<th>Mean posterior SD</th></tr>" + "".join(rows) + "</table>" + "".join(diagnostics)
        )


def simulate_toxfinder_native(
    dose_levels: ArrayLike,
    true_surface: ArrayLike,
    *,
    prior: ToxFinderPrior,
    target: float,
    stage1_patients: int,
    stage2_patients: int,
    cohort_size: int,
    trials: int,
    seed: int,
    start_index: int = 0,
    potency: float = 1.0,
    mode: Literal["fi_ckp", "fi"] = "fi_ckp",
    final_policy: Literal["legacy", "refit"] = "legacy",
    draws: int = 500,
    warmup: int = 250,
    chains: int = 2,
    max_sampler_evaluations: int = 20_000_000,
) -> ToxFinderNativeSimulation:
    """Repeated complete trials with independent streams and across-trial MCSE.

    At least two trials are required for sample SD/MCSE. Work and retained
    posterior arrays have explicit total bounds; each trial keeps its final fit.
    """
    count = _integer(trials, "trials", 2, 500)
    seed = _integer(seed, "seed", 0, 2**63 - 1)
    budget = _integer(max_sampler_evaluations, "max_sampler_evaluations", 1, 200_000_000)
    _integer(draws, "draws", 8, 10_000)
    _integer(chains, "chains", 2, 4)
    _integer(warmup, "warmup", 0, 100_000)
    cohort = _integer(cohort_size, "cohort_size", 1, 100)
    total_patients = _integer(stage1_patients, "stage1_patients", 1, 100) + _integer(
        stage2_patients, "stage2_patients", 0, 100
    )
    retained = count * chains * draws * (7 + 2 * total_patients)
    if retained > 2_000_000:
        raise ValueError(
            f"retained final fits require {retained} numeric values, exceeding the "
            "2000000-value cap. Run smaller batches with distinct seeds and save "
            "their results separately; changing sampler quality is not required."
        )
    if not isinstance(prior, ToxFinderPrior):
        raise TypeError("prior must be a ToxFinderPrior")
    _check_sampler_budget(
        count
        * _minimum_sampler_work(
            prior, stage1_patients, stage2_patients, cohort, draws, warmup, chains, final_policy
        ),
        budget,
    )
    results = []
    remaining = budget
    for stream in np.random.SeedSequence(seed).spawn(count):
        trial_seed = int(stream.generate_state(1, dtype=np.uint64)[0]) % (2**63)
        result = simulate_toxfinder_native_trial(
            dose_levels,
            true_surface,
            prior=prior,
            target=target,
            stage1_patients=stage1_patients,
            stage2_patients=stage2_patients,
            cohort_size=cohort_size,
            seed=trial_seed,
            start_index=start_index,
            potency=potency,
            mode=mode,
            final_policy=final_policy,
            draws=draws,
            warmup=warmup,
            chains=chains,
            max_sampler_evaluations=remaining,
        )
        remaining -= result.sampler_evaluations
        results.append(result)
    toxicities = np.array([r.toxicities for r in results])
    exposure = np.array([r.mean_true_exposure for r in results])
    pairs = np.array([r.final_doses for r in results])
    estimated = np.array([r.estimated_toxicity for r in results])
    truth = np.array([r.true_final_toxicity for r in results])
    return ToxFinderNativeSimulation(
        tuple(results),
        float(toxicities.mean()),
        float(toxicities.std(ddof=1)),
        float(toxicities.std(ddof=1) / np.sqrt(count)),
        float(exposure.mean()),
        float(exposure.std(ddof=1)),
        _frozen(pairs.mean(axis=0)),
        _frozen(pairs.std(axis=0, ddof=1)),
        _frozen(estimated.mean(axis=0)),
        _frozen(truth.mean(axis=0)),
        _frozen(abs(estimated - truth).mean(axis=0)),
        _frozen(np.mean([r.posterior_sd_toxicity for r in results], axis=0)),
    )
