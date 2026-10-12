"""Preflighted saved-scenario BOIN batches with distinct reproducible seeds."""

from dataclasses import dataclass

import numpy as np

from .boin import _owned
from .boin_desktop import BOINDesktopStudy
from .boin_desktop_simulation import (
    BOINDesktopSimulation,
    _validate_request,
    simulate_boin_desktop_combination,
)


@dataclass(frozen=True)
class BOINDesktopBatch:
    """Independent batches; legacy cross-trial defects restart at each batch."""

    results: tuple[BOINDesktopSimulation, ...]
    master_seed: int

    def aggregate(self, scenario_index: int) -> dict[str, np.ndarray | int]:
        runs = tuple(r for r in self.results if r.scenario_index == scenario_index)
        if not runs:
            raise ValueError("scenario_index is not present in this batch")
        total = sum(r.patients.shape[0] for r in runs)
        p = sum(r.selected.sum(axis=0) for r in runs) / total
        return {
            "trials": total,
            "selection_probability": _owned(p),
            "selection_mcse": _owned(np.sqrt(p * (1 - p) / total)),
            **{
                "mean_" + field: _owned(sum(getattr(r, field).sum(axis=0) for r in runs) / total)
                for field in ("patients", "toxicities", "actual_patients", "actual_toxicities")
            },
        }


def simulate_boin_desktop_scenarios(
    study: BOINDesktopStudy,
    *,
    scenario_indices: tuple[int, ...] | None = None,
    batches: int = 1,
    trials: int | None = None,
    seed: int | None = None,
    profile: str = "safe",
) -> BOINDesktopBatch:
    """Run saved scenarios after validating the complete allocation/work budget.

    At most 100 runs, two million retained trial-dose cells and one million
    planned patient draws across the request. Every run uses a distinct seed;
    the seed and resulting full arrays are retained for independent replay.
    """
    indices = (
        tuple(range(len(study.scenarios))) if scenario_indices is None else tuple(scenario_indices)
    )
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("scenario_indices must be nonempty and unique")
    if isinstance(batches, bool) or not isinstance(batches, int) or not 1 <= batches <= 100:
        raise ValueError("batches must be an integer in [1,100]")
    if len(indices) * batches > 100:
        raise ValueError("at most 100 scenario-batch runs are supported")
    validated = [_validate_request(study, i, trials, seed, profile) for i in indices]
    repetitions, master_seed = validated[0]
    runs = len(indices) * batches
    maximum = sum(study.subtrial_maximums) if study.mode == "contour" else study.maximum_patients
    overhead = (
        sum(study.shape) + study.cohort_size
        if profile == "desktop_legacy" and study.titration
        else 0
    )
    if repetitions * runs * np.prod(study.shape) > 2_000_000:
        raise ValueError("complete request exceeds 2000000 retained trial-dose cells")
    if repetitions * runs * (maximum + overhead) > 1_000_000:
        raise ValueError("complete request exceeds 1000000 planned patient draws")
    # Consecutive uint64 seeds are distinct, including at the wrap boundary.
    seeds = tuple((master_seed + i) % 2**64 for i in range(runs))
    results = tuple(
        simulate_boin_desktop_combination(
            study,
            scenario_index=scenario,
            trials=repetitions,
            seed=seeds[position * batches + batch],
            profile=profile,
        )
        for position, scenario in enumerate(indices)
        for batch in range(batches)
    )
    return BOINDesktopBatch(results, master_seed)
