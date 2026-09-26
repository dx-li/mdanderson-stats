"""All two-patient histories checked against independent base-R quadrature."""

import csv
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats.bmacrm import fit_bmacrm
from mdanderson_stats.crm_trial import run_crm_trial


def test_complete_trial_histories_against_independent_r_reference():
    with (Path(__file__).parent / "fixtures" / "crm-trial-reference.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4
    for row in rows:
        # Both doses carry the same potential outcome for this tiny enumeration.
        # A two-unit arrival gap exceeds the one-unit ascertainment window.
        delays = np.array([[0 if int(row[name]) else np.inf] * 2 for name in ("first", "second")])
        trial = run_crm_trial(
            [0.1, 0.3], [2, 2], delays, 1, target=0.25, cohort_size=1, safety_cutoff=1
        )
        assert_array_equal(trial.assigned_doses, [0, int(row["next_dose"])])
        assert_array_equal(trial.treated_counts, [int(row["n0"]), int(row["n1"])])
        assert_array_equal(trial.toxicities, [int(row["y0"]), int(row["y1"])])
        assert trial.selected_dose == int(row["selected_dose"])
        assert trial.steps[-1].high_uncertainty
        posterior = fit_bmacrm([0.1, 0.3], trial.toxicities, trial.treated_counts, target=0.25)
        assert_allclose(
            posterior.dose_mean, [float(row["mean0"]), float(row["mean1"])], atol=2e-10, rtol=0
        )
