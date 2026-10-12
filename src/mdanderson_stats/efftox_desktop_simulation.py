"""Current binary EffTox complete-outcome simulations with final reanalysis."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np
from numpy.typing import NDArray

from .efftox_desktop import EffToxDesktopStudy
from .efftox_desktop_controller import Analysis
from .efftox_desktop_decision import EffToxDesktopDecision
from .efftox_desktop_inference import EffToxDesktopFitDiagnostic, EffToxDesktopInference
from .efftox_model import _integer, _owned


@dataclass(frozen=True)
class EffToxDesktopSimulation:
    """Per-trial records; selected dose -1 denotes no selection (otherwise zero based).

    Rselect is conditional on a selection, as in the current desktop. It is None
    for a constant true-utility range, where the original divides by zero. It is
    zero when no trial selected a dose, matching the explicit native branch.
    Monte Carlo selection errors describe trial replication, not posterior error.
    """

    study: EffToxDesktopStudy
    scenario_index: int
    seed: int
    counts: NDArray[np.int64]
    selected_dose_index: NDArray[np.int64]
    stopped_early: NDArray[np.bool_]
    analyses_per_trial: NDArray[np.int64]
    true_utility: NDArray[np.float64]
    selection_probability: NDArray[np.float64]
    selection_mcse: NDArray[np.float64]
    mean_patients_per_dose: NDArray[np.float64]
    no_selection_probability: float
    rselect: float | None
    prior_decision: EffToxDesktopDecision
    inference_settings: tuple[tuple[str, str], ...]
    inference_diagnostics: tuple[EffToxDesktopFitDiagnostic, ...]


def _rselect(utilities: NDArray[np.float64], selected: NDArray[np.int64]) -> float | None:
    valid = selected[selected >= 0]
    if not valid.size:
        return 0.0
    span = float(np.max(utilities) - np.min(utilities))
    if span == 0:
        return None
    # Normalize before averaging to avoid overflow/cancellation in large sums.
    return float(np.mean((utilities[valid] - np.min(utilities)) / span))


def simulate_efftox_desktop(
    study: EffToxDesktopStudy,
    analyze: Analysis,
    *,
    scenario_index: int,
    trials: int,
    seed: int,
) -> EffToxDesktopSimulation:
    """Simulate from an empty history using a supplied current decision service.

    Analyze before each cohort and always once after stopping, including a final
    partial cohort. The same current-desktop reachability rule applies to final
    selection. Saved patients are not reused as simulated observations. Outcomes
    use the saved joint distribution; original RNG identity is not claimed.
    """
    scenario_id = _integer(scenario_index, "scenario_index", 0, len(study.scenarios) - 1)
    repetitions = _integer(trials, "trials", 1, 2000)
    random_seed = _integer(seed, "seed", 0, 2**63 - 1)
    if repetitions * len(study.doses) * 4 > 1_000_000:
        raise ValueError("simulation output exceeds 1000000 cells")
    diagnostic_start = (
        len(analyze.diagnostics) if isinstance(analyze, EffToxDesktopInference) else 0
    )
    settings: tuple[tuple[str, str], ...] = ()
    if isinstance(analyze, EffToxDesktopInference):
        analyze.check_study(study)
        settings = analyze.configuration
        analyze.require_analyses(
            1 + repetitions * (ceil(study.max_patients / study.cohort_size) + 1)
        )
    scenario = study.scenarios[scenario_id]
    truth = scenario.joint_probabilities()
    utility = study.contour().utility(np.array(scenario.efficacy), np.array(scenario.toxicity))
    rng = np.random.default_rng(random_seed)
    counts = np.zeros((repetitions, len(study.doses), 2, 2), dtype=np.int64)
    selected = np.full(repetitions, -1, dtype=np.int64)
    early = np.zeros(repetitions, dtype=bool)
    calls = np.zeros(repetitions, dtype=np.int64)

    def evaluate(matrix: NDArray[np.int64]) -> EffToxDesktopDecision:
        protected = matrix.copy()
        protected.flags.writeable = False
        result = analyze(protected)
        if not isinstance(result, EffToxDesktopDecision):
            raise TypeError("analysis service must return EffToxDesktopDecision")
        if result.dose_index is not None:
            _integer(result.dose_index, "selected dose", 0, len(study.doses) - 1)
        return result

    prior = evaluate(np.zeros(counts.shape[1:], dtype=np.int64))
    for trial in range(repetitions):
        enrolled = 0
        while enrolled < study.max_patients:
            decision = evaluate(counts[trial])
            calls[trial] += 1
            dose = decision.dose_index
            if dose is None:
                early[trial] = True
                break
            size = min(study.cohort_size, study.max_patients - enrolled)
            cells = rng.multinomial(size, truth[dose].reshape(4))
            counts[trial, dose] += cells.reshape(2, 2)
            enrolled += size
        final = evaluate(counts[trial])
        calls[trial] += 1
        if final.dose_index is None:
            early[trial] = True
        else:
            selected[trial] = final.dose_index
    probabilities = np.array([(selected == dose).mean() for dose in range(len(study.doses))])
    for array in (counts, selected, early, calls):
        array.flags.writeable = False
    return EffToxDesktopSimulation(
        study,
        scenario_id,
        random_seed,
        counts,
        selected,
        early,
        calls,
        _owned(utility),
        _owned(probabilities),
        _owned(np.sqrt(probabilities * (1 - probabilities) / repetitions)),
        _owned(counts.sum(axis=(2, 3)).mean(axis=0)),
        float(np.mean(selected == -1)),
        _rselect(utility, selected),
        prior,
        settings,
        analyze.diagnostics[diagnostic_start:]
        if isinstance(analyze, EffToxDesktopInference)
        else (),
    )
