"""All-scenario execution and auditable aggregation of EffTox trial batches."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil

import numpy as np
from numpy.typing import NDArray

from .efftox_desktop import EffToxDesktopStudy
from .efftox_desktop_controller import Analysis
from .efftox_desktop_inference import EffToxDesktopInference
from .efftox_desktop_report import _diagnostics, _page, _table
from .efftox_desktop_simulation import (
    EffToxDesktopSimulation,
    _rselect,
    simulate_efftox_desktop,
)
from .efftox_model import _integer, _owned


@dataclass(frozen=True)
class EffToxDesktopBatchSummary:
    """Trial-weighted summaries retaining every original batch and random seed."""

    batches: tuple[EffToxDesktopSimulation, ...]
    trials: int
    selection_probability: NDArray[np.float64]
    selection_mcse: NDArray[np.float64]
    mean_patients_per_dose: NDArray[np.float64]
    no_selection_probability: float
    no_selection_mcse: float
    early_stop_probability: float
    rselect: float | None


def summarize_efftox_desktop_batches(
    batches: Sequence[EffToxDesktopSimulation],
) -> EffToxDesktopBatchSummary:
    """Combine unequal-size, distinct-outcome-seed batches of the same scenario.

    Saved studies and recorded statistical/sampler settings must agree, except
    sampler seed and work budget. Custom-service identity cannot be inferred
    when settings were not supplied. Seed uniqueness prevents exact stream reuse;
    it is not a proof of statistical independence of user-supplied services.
    """
    values = tuple(batches)
    if not 1 <= len(values) <= 100 or not all(
        isinstance(v, EffToxDesktopSimulation) for v in values
    ):
        raise ValueError("supply between 1 and 100 EffTox simulation batches")
    first = values[0]

    def settings(batch: EffToxDesktopSimulation) -> dict[str, str]:
        return {k: v for k, v in batch.inference_settings if k not in {"sampler_seed", "max_work"}}

    for batch in values:
        if batch.study != first.study or batch.scenario_index != first.scenario_index:
            raise ValueError("batches must have the same saved study and scenario")
        if settings(batch) != settings(first):
            raise ValueError("batches have different recorded inference settings")
    if len({b.seed for b in values}) != len(values):
        raise ValueError("batch outcome seeds must be distinct")
    trials = sum(len(b.selected_dose_index) for b in values)
    if not 1 <= trials <= 100_000:
        raise ValueError("combined trial count must be between 1 and 100000")
    selected = np.concatenate([b.selected_dose_index for b in values])
    probability = np.array([(selected == i).mean() for i in range(len(first.study.doses))])
    patients = sum((b.counts.sum(axis=(0, 2, 3)) for b in values), np.zeros(len(probability)))
    none = float(np.mean(selected == -1))
    early = sum(int(b.stopped_early.sum()) for b in values) / trials
    return EffToxDesktopBatchSummary(
        values,
        trials,
        _owned(probability),
        _owned(np.sqrt(probability * (1 - probability) / trials)),
        _owned(patients / trials),
        none,
        float(np.sqrt(none * (1 - none) / trials)),
        early,
        _rselect(first.true_utility, selected),
    )


def simulate_efftox_desktop_scenarios(
    study: EffToxDesktopStudy,
    analyze: Analysis,
    *,
    trials: int | None = None,
    seed: int | None = None,
) -> tuple[EffToxDesktopSimulation, ...]:
    """Run every saved scenario, using saved QtyReps/Seed unless overridden.

    Independent derived outcome streams are recorded in every result. This is
    reproducible Python RNG behavior, not native RNG identity. The built-in
    inference service preflights the entire request before executing any scenario.
    """
    if not study.scenarios:
        raise ValueError("the saved study has no simulation scenarios")
    simulation = ET.fromstring(study.xml).find("SimulationParams")
    assert simulation is not None

    def saved(name: str) -> int:
        try:
            return int(simulation.findtext(name, ""))
        except ValueError as exc:
            raise ValueError(f"saved {name} must be an integer or supplied explicitly") from exc

    repetitions = _integer(saved("QtyReps") if trials is None else trials, "trials", 1, 2000)
    random_seed = _integer(saved("Seed") if seed is None else seed, "seed", 0, 2**63 - 1)
    if isinstance(analyze, EffToxDesktopInference):
        analyze.check_study(study)
        analyze.require_analyses(
            len(study.scenarios)
            * (1 + repetitions * (ceil(study.max_patients / study.cohort_size) + 1))
        )
    streams = np.random.SeedSequence(random_seed).spawn(len(study.scenarios))
    return tuple(
        simulate_efftox_desktop(
            study,
            analyze,
            scenario_index=index,
            trials=repetitions,
            seed=int(stream.generate_state(1, dtype=np.uint64)[0]) & (2**63 - 1),
        )
        for index, stream in enumerate(streams)
    )


def efftox_desktop_batches_html(batches: Sequence[EffToxDesktopSimulation]) -> str:
    """Report all supplied scenarios/batches with weighted OCs and original seeds.

    Prior-only summaries remain separately labeled per batch. They are not
    averaged as if each batch had identical posterior precision or trial counts.
    """
    values = tuple(batches)
    if not 1 <= len(values) <= 100 or not all(
        isinstance(b, EffToxDesktopSimulation) for b in values
    ):
        raise ValueError("supply between 1 and 100 EffTox simulation batches")
    study = values[0].study
    if any(b.study != study for b in values):
        raise ValueError("reported batches must have the same saved study")
    groups = {
        i: [b for b in values if b.scenario_index == i]
        for i in sorted({b.scenario_index for b in values})
    }
    body = (
        f"<p>Included scenario indices: {list(groups)}; "
        f"saved scenarios: {len(study.scenarios)}.</p>"
    )
    for index, group in groups.items():
        summary = summarize_efftox_desktop_batches(group)
        scenario = study.scenarios[index]
        body += f"<h2>Scenario {index}: {summary.trials} trials</h2>" + _table(
            (
                "Dose",
                "True efficacy",
                "True toxicity",
                "True utility",
                "Selection probability",
                "Selection MCSE",
                "Mean patients",
            ),
            [
                (
                    dose,
                    scenario.efficacy[i],
                    scenario.toxicity[i],
                    group[0].true_utility[i],
                    summary.selection_probability[i],
                    summary.selection_mcse[i],
                    summary.mean_patients_per_dose[i],
                )
                for i, dose in enumerate(study.doses)
            ],
        )
        body += _table(
            ("Aggregate field", "Value"),
            [
                ("No-selection probability", summary.no_selection_probability),
                ("No-selection MCSE", summary.no_selection_mcse),
                ("Early-stop flag probability", summary.early_stop_probability),
                (
                    "Rselect conditional on selection",
                    summary.rselect
                    if summary.rselect is not None
                    else "undefined: constant true utility",
                ),
            ],
        )
        for number, batch in enumerate(group, 1):
            body += f"<h3>Scenario {index}, batch {number}</h3>" + _table(
                ("Batch setting", "Value"),
                [
                    ("Outcome seed", batch.seed),
                    ("Trials", len(batch.selected_dose_index)),
                    *batch.inference_settings,
                ],
            )
            prior = batch.prior_decision
            means = batch.inference_diagnostics[0] if batch.inference_diagnostics else None
            body += "<h4>Prior-only diagnostics for this batch</h4>" + _table(
                (
                    "Dose",
                    "Mean efficacy",
                    "Mean toxicity",
                    "Efficacy tail",
                    "Toxicity tail",
                    "Utility",
                ),
                [
                    (
                        dose,
                        means.mean_efficacy[i] if means else "not supplied",
                        means.mean_toxicity[i] if means else "not supplied",
                        prior.efficacy_tail_probability[i],
                        prior.toxicity_tail_probability[i],
                        prior.utility[i],
                    )
                    for i, dose in enumerate(study.doses)
                ],
            )
            body += _diagnostics(batch.inference_diagnostics)
            if len(body) > 50_000_000:
                raise ValueError("batch report exceeds 50000000 characters; report fewer batches")
    return _page("EffTox scenario and batch report", body, study)
