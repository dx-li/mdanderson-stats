"""Historical-control allocation compared with R and original Fortran."""

import csv
from pathlib import Path

import numpy as np

from mdanderson_stats.stplan_historical_planning import stplan_historical_allocation_plan
from mdanderson_stats.stplan_survival import stplan_historical_survival_power


def test_historical_allocation_against_independent_and_native_references():
    fixtures = Path(__file__).parent / "fixtures"
    with (fixtures / "stplan-historical-allocation-native.csv").open(newline="") as handle:
        native = {row["case_id"]: row for row in csv.DictReader(handle)}
    with (fixtures / "stplan-historical-allocation-r.csv").open(newline="") as handle:
        independent = list(csv.DictReader(handle))
    for row in independent:
        fixed = {
            name: float(row[name])
            for name in (
                "experimental_hazard",
                "control_hazard",
                "accrual_rate",
                "followup_duration",
                "historical_deaths",
                "historical_alive",
                "alpha",
            )
        }
        fixed["continued_followup"] = row["continued_followup"] == "TRUE"
        plan = stplan_historical_allocation_plan(
            **fixed,
            target_power=float(row["target"]),
            accrual_bounds=(float(row["accrual_lower"]), float(row["accrual_upper"])),
            allocation_bounds=(float(row["allocation_lower"]), float(row["allocation_upper"])),
        )
        np.testing.assert_allclose(
            plan.accrual_duration, float(row["accrual_duration"]), rtol=2e-7, atol=1e-8
        )
        np.testing.assert_allclose(
            plan.control_allocation, float(row["allocation"]), rtol=0, atol=2e-6
        )
        np.testing.assert_allclose(plan.achieved_power, float(row["achieved"]), rtol=0, atol=1e-8)
        np.testing.assert_allclose(
            stplan_historical_survival_power(**plan.inputs), plan.achieved_power, rtol=0, atol=1e-14
        )
        if row["case_id"] in native:
            original = native[row["case_id"]]
            assert original["status"] == "0" and original["ok"] == "T"
            np.testing.assert_allclose(
                plan.accrual_duration, float(original["accrual_duration"]), rtol=2e-7, atol=1e-8
            )
            np.testing.assert_allclose(
                plan.control_allocation, float(original["allocation"]), rtol=0, atol=2e-6
            )
