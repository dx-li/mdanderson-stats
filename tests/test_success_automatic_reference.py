import csv
from pathlib import Path

import numpy as np
import pytest
from mdanderson_stats.success_calibration_binary_search import calibrate_binary_success_cutoff
from mdanderson_stats.success_calibration_continuous import (
    calibrate_normal_success_cutoff,
    calibrate_survival_success_cutoff,
)

from mdanderson_stats.success_calibration import binary_success_oc, normal_success_oc

FIXTURES = Path(__file__).parent / "fixtures"


def _rows(name: str) -> list[dict[str, str]]:
    with (FIXTURES / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def test_binary_strict_breakpoint_states_match_independent_r_reference():
    settings = {r["case"]: r for r in _rows("success-automatic-binary-settings.csv")}
    selected = {r["case"]: r for r in _rows("success-automatic-binary-selected.csv")}
    state_rows = _rows("success-automatic-binary-interior-probes.csv")
    for name, spec in settings.items():
        kwargs = dict(
            n=int(spec["n"]),
            margin=_f(spec, "margin"),
            design_prior=(_f(spec, "design_alpha"), _f(spec, "design_beta")),
            analysis_prior=(_f(spec, "analysis_alpha"), _f(spec, "analysis_beta")),
            direction=spec["direction"],
        )
        if spec["feasible"] == "TRUE":
            calibrated = calibrate_binary_success_cutoff(
                target=_f(spec, "target_pid"),
                cutoff_range=(_f(spec, "cutoff_min"), _f(spec, "cutoff_max")),
                **kwargs,
            )
            expected = selected[name]
            assert calibrated.cutoff == pytest.approx(_f(expected, "cutoff"), abs=2e-14)
            assert (
                calibrated.operating_characteristics.incorrect_decision_probability
                == pytest.approx(_f(expected, "pid"), abs=2e-13)
            )
            assert calibrated.operating_characteristics.true_positive == pytest.approx(
                _f(expected, "true_positive"), abs=2e-13
            )
            assert calibrated.operating_characteristics.false_positive == pytest.approx(
                _f(expected, "false_positive"), abs=2e-13
            )
            assert 1 <= calibrated.candidates_evaluated <= int(spec["n"]) + 2
            if name == "exact_n1":
                assert calibrated.cutoff == pytest.approx(0.5)
        else:
            with pytest.raises(ValueError, match="no cutoff"):
                calibrate_binary_success_cutoff(
                    target=_f(spec, "target_pid"),
                    cutoff_range=(_f(spec, "cutoff_min"), _f(spec, "cutoff_max")),
                    **kwargs,
                )

    # Check independent operating characteristics inside every breakpoint
    # interval. This avoids making cross-language equality at a beta-tail
    # breakpoint depend on the final ulp of pbeta versus SciPy special functions.
    for row in state_rows:
        spec = settings[row["case"]]
        observed = binary_success_oc(
            int(spec["n"]),
            _f(row, "cutoff"),
            margin=_f(spec, "margin"),
            design_prior=(_f(spec, "design_alpha"), _f(spec, "design_beta")),
            analysis_prior=(_f(spec, "analysis_alpha"), _f(spec, "analysis_beta")),
            direction=spec["direction"],
        )
        assert observed.true_positive == pytest.approx(_f(row, "true_positive"), abs=3e-13)
        assert observed.false_positive == pytest.approx(_f(row, "false_positive"), abs=3e-13)
        assert observed.bayesian_power == pytest.approx(_f(row, "success_probability"), abs=3e-13)
        if row["pid"] == "NA":
            assert observed.incorrect_decision_probability is None
        else:
            assert observed.incorrect_decision_probability == pytest.approx(
                _f(row, "pid"), abs=3e-13
            )

    # With one observation and uniform analysis prior, the zero-response
    # posterior tail is .25. Strict `>` excludes that state at the breakpoint;
    # the immediately lower cutoff includes it.
    tied = binary_success_oc(1, 0.25, margin=0.5)
    below = binary_success_oc(1, np.nextafter(0.25, 0.0), margin=0.5)
    assert tied.true_positive == pytest.approx(3 / 8)
    assert tied.false_positive == pytest.approx(1 / 8)
    assert below.bayesian_power == pytest.approx(1.0)


def test_normal_and_survival_bisection_match_independent_r_integrals():
    settings = {r["case"]: r for r in _rows("success-automatic-normal-settings.csv")}
    selected = {r["case"]: r for r in _rows("success-automatic-normal-selected.csv")}
    curve_rows = _rows("success-automatic-normal-curve.csv")
    calibrated_results = {}
    for name, spec in settings.items():
        arms = int(spec["n_arms"])
        kwargs = {
            "standard_error": [_f(spec, f"se_{i}") for i in range(1, arms + 1)]
            if arms == 2
            else _f(spec, "se_1"),
            "design_mean": [_f(spec, f"design_mean_{i}") for i in range(1, arms + 1)]
            if arms == 2
            else _f(spec, "design_mean_1"),
            "design_sd": [_f(spec, f"design_sd_{i}") for i in range(1, arms + 1)]
            if arms == 2
            else _f(spec, "design_sd_1"),
            "analysis_mean": [_f(spec, f"analysis_mean_{i}") for i in range(1, arms + 1)]
            if arms == 2
            else _f(spec, "analysis_mean_1"),
            "analysis_sd": [_f(spec, f"analysis_sd_{i}") for i in range(1, arms + 1)]
            if arms == 2
            else _f(spec, "analysis_sd_1"),
            "margin": _f(spec, "margin"),
            "direction": spec["direction"],
            "null_mean": [_f(spec, f"null_mean_{i}") for i in range(1, arms + 1)]
            if arms == 2
            else _f(spec, "null_mean_1"),
        }
        result = calibrate_normal_success_cutoff(
            _f(spec, "target_pid"),
            cutoff_range=(_f(spec, "cutoff_min"), _f(spec, "cutoff_max")),
            cutoff_tolerance=1e-8,
            **kwargs,
        )
        calibrated_results[name] = result
        ref = selected[name]
        assert result.bracket[0] <= result.bracket[1]
        assert result.bracket[1] - result.bracket[0] <= result.cutoff_tolerance
        assert result.cutoff == result.bracket[1]
        assert result.cutoff == pytest.approx(_f(ref, "upper_feasible"), abs=3e-8)
        assert result.operating_characteristics.incorrect_decision_probability == pytest.approx(
            _f(ref, "upper_pid"), abs=3e-8
        )
        assert result.operating_characteristics.incorrect_decision_probability <= result.target

    for row in curve_rows:
        spec = settings[row["case"]]
        arms = int(spec["n_arms"])
        observed = normal_success_oc(
            _f(row, "cutoff"),
            standard_error=[_f(spec, f"se_{i}") for i in range(1, arms + 1)]
            if arms == 2
            else _f(spec, "se_1"),
            design_mean=[_f(spec, f"design_mean_{i}") for i in range(1, arms + 1)]
            if arms == 2
            else _f(spec, "design_mean_1"),
            design_sd=[_f(spec, f"design_sd_{i}") for i in range(1, arms + 1)]
            if arms == 2
            else _f(spec, "design_sd_1"),
            analysis_mean=[_f(spec, f"analysis_mean_{i}") for i in range(1, arms + 1)]
            if arms == 2
            else _f(spec, "analysis_mean_1"),
            analysis_sd=[_f(spec, f"analysis_sd_{i}") for i in range(1, arms + 1)]
            if arms == 2
            else _f(spec, "analysis_sd_1"),
            margin=_f(spec, "margin"),
            direction=spec["direction"],
            null_mean=[_f(spec, f"null_mean_{i}") for i in range(1, arms + 1)]
            if arms == 2
            else _f(spec, "null_mean_1"),
        )
        assert observed.incorrect_decision_probability == pytest.approx(_f(row, "pid"), abs=2e-10)
        assert observed.bayesian_power == pytest.approx(_f(row, "success_probability"), abs=2e-10)
        assert observed.true_positive == pytest.approx(_f(row, "true_positive"), abs=2e-10)
        assert observed.false_positive == pytest.approx(_f(row, "false_positive"), abs=2e-10)

    reflected = calibrated_results["normal_reflected_less"].operating_characteristics
    original = calibrated_results["normal_unequal_two_arm"].operating_characteristics
    assert reflected.true_positive == pytest.approx(original.true_positive, abs=2e-13)
    assert reflected.false_positive == pytest.approx(original.false_positive, abs=2e-13)
    assert reflected.bayesian_power == pytest.approx(original.bayesian_power, abs=2e-13)

    survival_settings = _rows("success-automatic-survival-settings.csv")
    survival_selected = {r["case"]: r for r in _rows("success-automatic-survival-selected.csv")}
    survival_results = {}
    for spec in survival_settings:
        name = spec["case"]
        result = calibrate_survival_success_cutoff(
            _f(spec, "target_pid"),
            cutoff_range=(_f(spec, "cutoff_min"), _f(spec, "cutoff_max")),
            cutoff_tolerance=1e-8,
            events=_f(spec, "events"),
            treatment_allocation=_f(spec, "treatment_allocation"),
            design_mean=_f(spec, "design_mean"),
            design_sd=_f(spec, "design_sd"),
            analysis_mean=_f(spec, "analysis_mean"),
            analysis_sd=_f(spec, "analysis_sd"),
            margin=_f(spec, "margin"),
        )
        ref = survival_selected[name]
        assert result.cutoff == pytest.approx(_f(ref, "upper_feasible"), abs=3e-8)
        assert result.operating_characteristics.incorrect_decision_probability == pytest.approx(
            _f(ref, "upper_pid"), abs=3e-8
        )
        survival_results[name] = result
    assert survival_results["survival_base"].cutoff == pytest.approx(
        survival_results["survival_rescaled"].cutoff, abs=2e-14
    )


def test_normal_calibration_endpoint_and_unmet_target_contract():
    model = dict(
        standard_error=0.45,
        design_mean=0.10,
        design_sd=0.38,
        analysis_mean=-0.05,
        analysis_sd=0.72,
        margin=0.15,
    )
    endpoint_pid = normal_success_oc(0.6, **model).incorrect_decision_probability
    assert endpoint_pid is not None
    endpoint = calibrate_normal_success_cutoff(endpoint_pid, cutoff_range=(0.6, 0.999), **model)
    assert endpoint.cutoff == 0.6
    assert endpoint.bracket == (0.6, 0.6)
    assert endpoint.candidates_evaluated == 1

    upper_pid = normal_success_oc(0.999, **model).incorrect_decision_probability
    assert upper_pid is not None and upper_pid > 0
    with pytest.raises(ValueError, match="upper cutoff"):
        calibrate_normal_success_cutoff(upper_pid / 2, cutoff_range=(0.6, 0.999), **model)
