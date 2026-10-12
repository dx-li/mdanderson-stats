"""Actual precision-qualified calendar run; a computational check, not design validation."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from mdanderson_stats import (
    UAROETDesktopScenario,
    read_uaroet_desktop_study,
    run_uaroet_desktop_trial,
)

study = read_uaroet_desktop_study(
    (Path(__file__).resolve().parents[1] / "tests/fixtures/uaroet-python-study.json").read_text()
)
p = dict(study.trial_parameters)
p.update(
    NumPatients=3,
    CohortSize_First=1,
    NumTimesToShrinkUtilityCutoff=1,
    MinProbUtilityIsHighest=0,
    UpperAcceptablePiToxCutoff=1,
    MeanDaysBetweenArrivals=0.1,
    MinDaysToEfficacy=1,
    MaxDaysToEfficacy=1,
    MinDaysToToxicity=1,
    MaxDaysToToxicity=1,
)
study = replace(study, patients=(), trial_parameters=p)
scenario = UAROETDesktopScenario(1, [[0.7, 0.3], [0.5, 0.5]], [[0.9, 0.1], [0.7, 0.3]], 0.2)
looks = []


def record(clock, decision):
    fit = decision.posterior
    looks.append(
        dict(
            clock=clock,
            action=decision.action,
            reason=decision.reason,
            final=decision.final,
            observed_rows=decision.study.patients,
            selected=decision.selected,
            draws_per_chain=None if fit is None else fit.draws_per_chain,
            maximum_mcse_ratio=None if fit is None else float(np.max(fit.utility_mcse_ratio)),
            maximum_rhat=None if fit is None else float(np.max(fit.utility_split_rhat)),
        )
    )


trial = run_uaroet_desktop_trial(
    study,
    scenario,
    rng=np.random.default_rng(1092),
    max_draws=50000,
    max_evaluations=8000000,
    on_look=record,
)
print(
    json.dumps(
        dict(
            scope="three-patient pending-outcome calendar computational check",
            seed=1092,
            target=0.03,
            maximum_rhat=1.05,
            looks=looks,
            enrollment=len(trial.patients),
            selected=trial.selected,
            followup_duration=trial.followup_duration,
            decision_time=trial.decision_time,
            clinical_validation=False,
        ),
        indent=2,
        allow_nan=False,
    )
)
