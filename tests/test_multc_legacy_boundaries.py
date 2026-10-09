import hashlib
import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from mdanderson_stats import (
    multc_lean_design,
    multc_legacy_boundaries,
    run_multc_legacy_design_duration,
    run_multc_legacy_duration,
)

FIXTURES = Path(__file__).parent / "fixtures"
REFERENCE = json.loads((FIXTURES / "multc-lean-native-boundaries.json").read_text())


def design_for(case):
    r, t = (case[e]["parameters"] for e in ("response", "toxicity"))

    def history(row, complement=False):
        if row["h"]:
            return 1 - float(row["h"]) if complement else float(row["h"])
        a, b = float(row["ha"]), float(row["hb"])
        return (b, a) if complement else (a, b)

    return multc_lean_design(
        int(r["cap"]),
        (float(r["a"]), float(r["b"])),
        (float(t["b"]), float(t["a"])),
        historical_response=history(r),
        historical_toxicity=history(t, True),
        response_margin=float(r["delta"]),
        toxicity_margin=-float(t["delta"]),
        response_cutoff=float(r["cutoff"]),
        toxicity_cutoff=float(t["cutoff"]),
        min_subjects=int(r["minimum"]),
        cohort_size=int(r["cohort"]),
    )


@pytest.fixture(scope="module", params=REFERENCE["cases"], ids=lambda c: c["case"])
def native_case(request):
    case = request.param
    return case, design_for(case)


def test_base_r_probability_table_checksum():
    digest = hashlib.sha256((FIXTURES / "multc-lean-boundary-r-tails.csv").read_bytes()).hexdigest()
    assert digest == REFERENCE["r_table_sha256"]


def test_compact_vectors_match_original_control_instructions(native_case):
    case, design = native_case
    boundaries = multc_legacy_boundaries(design)
    assert boundaries.response_stop_at == tuple(case["response"]["vector"])
    assert boundaries.nontoxicity_stop_at == tuple(case["toxicity"]["vector"])
    assert boundaries.prior_response == (case["response"]["vector"] == [0])
    assert boundaries.prior_toxicity == (case["toxicity"]["vector"] == [0])
    assert boundaries.prior_rejected == (boundaries.prior_response or boundaries.prior_toxicity)
    assert boundaries.max_subjects == design.max_subjects
    assert boundaries.min_subjects == design.min_subjects
    assert boundaries.cohort_size == design.cohort_size


def test_posterior_predicates_match_independent_base_r(native_case):
    case, design = native_case
    for endpoint in ("response", "toxicity"):
        cutoff = getattr(design, endpoint + "_cutoff")
        for n, successes, probability in case[endpoint]["predicate_calls"]:
            events = successes if endpoint == "response" else n - successes
            actual = design._prob(endpoint, n, events)
            # The production comparison requests 1e-8 integration accuracy
            # away from a cutoff, refining only ambiguous stopping decisions.
            assert actual == pytest.approx(probability, rel=2e-8, abs=1e-8)
            assert design._stop(endpoint, actual) == (probability > cutoff)


def test_complement_mapping_matches_original_instructions(native_case):
    case, _ = native_case
    for endpoint in ("response", "toxicity"):
        row = case[endpoint]["parameters"]
        output = case[endpoint]["native_complement"]
        assert output[0] == bool(row["h"])
        assert output[1] == (1 - float(row["h"]) if row["h"] else 0)
        assert output[2:4] == ([0, 0] if row["h"] else [float(row["hb"]), float(row["ha"])])
        assert output[4:6] == [float(row["b"]), float(row["a"])]
        assert output[6:] == [float(row["cutoff"]), -float(row["delta"])]


def test_design_duration_composes_original_vectors_and_kernel(native_case):
    case, design = native_case
    for reference in case["duration_cases"]:
        inputs = reference["inputs"]
        trial = run_multc_legacy_design_duration(
            design,
            joint_probabilities=inputs["prob"],
            mean_interarrival=inputs["mean"],
            response_window=inputs["window"],
            uniforms=inputs["uniform"],
            unit_exponentials=inputs["unit_exponential"],
        )
        for key, value in reference["native"].items():
            assert getattr(trial, key) == pytest.approx(value, rel=2e-15, abs=1e-14)
        assert trial == run_multc_legacy_duration(
            design.max_subjects,
            response_stop_at=inputs["response"],
            nontoxicity_stop_at=inputs["notox"],
            joint_probabilities=inputs["prob"],
            mean_interarrival=inputs["mean"],
            response_window=inputs["window"],
            uniforms=inputs["uniform"],
            unit_exponentials=inputs["unit_exponential"],
        )


def basic_design(**changes):
    return multc_lean_design(
        12,
        (1, 1),
        (1, 1),
        historical_response=0.5,
        historical_toxicity=0.5,
        **changes,
    )


def trial_arguments(**changes):
    return {
        **dict(
            joint_probabilities=[0.25] * 4,
            mean_interarrival=1.0,
            response_window=2.0,
            uniforms=[],
            unit_exponentials=[],
        ),
        **changes,
    }


@pytest.mark.parametrize(
    "response,toxicity,reason",
    [
        (0.49, 0.95, "prior_response"),
        (0.95, 0.49, "prior_toxicity"),
        (0.49, 0.49, "prior_both"),
    ],
)
def test_native_prior_screen_precedes_minimum_and_consumes_no_draws(response, toxicity, reason):
    design = basic_design(response_cutoff=response, toxicity_cutoff=toxicity, min_subjects=12)
    trial = run_multc_legacy_design_duration(design, **trial_arguments())
    assert trial.decision == reason
    assert trial.sample_size == trial.responses == trial.toxicities == trial.balks == 0
    assert trial.duration == trial.decision_time == 0
    assert trial.uniforms_consumed == trial.exponentials_consumed == 0
    assert trial.patients == ()


@pytest.mark.parametrize(
    "change",
    [
        {"mean_interarrival": 0},
        {"response_window": float("nan")},
        {"joint_probabilities": [0.5] * 4},
        {"uniforms": [0]},
        {"unit_exponentials": [-1]},
        {"uniforms": [[0.2]]},
    ],
)
def test_prior_rejection_still_validates_inputs(change):
    with pytest.raises(ValueError):
        run_multc_legacy_design_duration(
            basic_design(response_cutoff=0.49), **trial_arguments(**change)
        )


def test_native_cap_placeholder_is_not_an_adverse_stop():
    design = basic_design(response_cutoff=1, toxicity_cutoff=1)
    bounds = multc_legacy_boundaries(design)
    assert bounds.response_stop_at == bounds.nontoxicity_stop_at == (12,)
    assert not bounds.prior_rejected
    trial = run_multc_legacy_design_duration(
        design, **trial_arguments(uniforms=[0.9] * 12, unit_exponentials=[0.1] * 11)
    )
    assert trial.sample_size == 12
    assert trial.decision == "cap_complete"
    assert trial.balks == 0
    assert trial.exponentials_consumed == 11


def test_native_profile_rejects_disabled_pretrial_screen_and_wrong_design_type():
    with pytest.raises(ValueError, match="pretrial_check=True"):
        multc_legacy_boundaries(basic_design(pretrial_check=False))
    with pytest.raises(ValueError, match="MultcLeanDesign"):
        multc_legacy_boundaries(None)


def test_boundary_result_is_immutable():
    bounds = multc_legacy_boundaries(basic_design())
    with pytest.raises(FrozenInstanceError):
        bounds.max_subjects = 24
    with pytest.raises(TypeError):
        bounds.response_stop_at[0] = 6
