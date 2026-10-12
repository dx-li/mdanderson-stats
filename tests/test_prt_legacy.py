import json
import math
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import (
    prt_legacy_beta_moments,
    prt_legacy_final_selection,
    prt_legacy_predictive,
    prt_legacy_projection,
    prt_legacy_screening,
    prt_legacy_simulation_decision,
)

ROOT = Path(__file__).parent / "fixtures"


def cases(name):
    return json.loads((ROOT / f"prt-desktop-{name}.json").read_text())["cases"]


@pytest.mark.parametrize("c", cases("projection"))
def test_projection(c):
    result = prt_legacy_projection(np.asarray(c["input"])[:, None, :])
    np.testing.assert_allclose(result.transformed[:, 0], c["result"], atol=1e-11, rtol=0)
    assert result.valid_probabilities == bool(np.max(c["result"]) <= 1)
    assert not result.transformed.flags.writeable


@pytest.mark.parametrize("c", json.loads((ROOT / "prt-desktop-moments.json").read_text())["cases"])
def test_moments(c):
    result = prt_legacy_beta_moments(c["mean"], c["sd"])
    i = c["current"] - 1
    for actual, expected in ((result.alpha[i], c["alpha"]), (result.beta[i], c["beta"])):
        if expected == "nan":
            assert math.isnan(actual)
        elif expected == "inf":
            assert actual == math.inf
        elif expected == "-inf":
            assert actual == -math.inf
        else:
            assert actual == pytest.approx(expected, abs=1e-12)
    if "draws" in c:
        x = np.asarray(c["draws"])
        np.testing.assert_allclose(x.mean(0), c["mean"], atol=2e-15)
        np.testing.assert_allclose(x.std(0), c["sd"], atol=2e-15)
        np.testing.assert_allclose((x > 0.3).mean(0), c["exceedance"], atol=2e-15)


@pytest.mark.parametrize("c", cases("predictive"))
def test_predictive(c):
    result = prt_legacy_predictive(
        c["alpha"],
        c["beta"],
        c["probabilities"],
        target=c["target"],
        negligible_cutoff=min(0.3, c["upper"] / 2),
        excessive_cutoff=c["upper"],
    )
    assert result.excessive == pytest.approx(c["predictive_excessive"], abs=1e-12)


@pytest.mark.parametrize("c", cases("simulation-rules"))
def test_simulation(c):
    result = prt_legacy_simulation_decision(
        c["exceedance"],
        c["distance"],
        c["current"] - 1,
        c["pending"],
        c["pn"],
        c["pe"],
        c["next_pe"],
    )
    assert result.source_dose == c["selected"] - 1
    assert (result.action == "stop") == bool(c["stopped"])
    assert (result.action == "suspend") == bool(c["suspend"])


@pytest.mark.parametrize("c", cases("final"))
def test_final(c):
    result = prt_legacy_final_selection(
        c["assessed"], c["exceedance"], c["distance"], c["current"] - 1
    )
    assert result.source_dose == c["selected"] - 1
    assert (result.action == "stop") == bool(c["stopped"])
    if result.action == "legacy_invalid_selection":
        assert result.warnings


@pytest.mark.parametrize("c", cases("screening"))
def test_screening(c):
    result = prt_legacy_screening(
        c["assigned"],
        c["assessed"],
        c["observed"],
        c["first_interval"],
        c["current"] - 1,
        final=c["final"],
    )
    assert (result.infer, result.suspend) == (c["infer"], bool(c["suspend"]))


@pytest.mark.parametrize("p", [[0.0], [1.0], [0.5, 0.0], [0.5, 1.0]])
def test_endpoint_jitter_is_not_a_valid_prediction(p):
    with pytest.raises(ValueError, match="interior"):
        prt_legacy_predictive(1.0, 3.0, p)


def test_invalid_native_moments_are_not_silently_repaired():
    r = prt_legacy_beta_moments([0.5, 0.5, 0.5], [0.0, 0.5, 0.7])
    assert np.isinf(r.alpha[0]) and r.alpha[1] == 0 and r.alpha[2] == float(np.float32(0.05))
    assert not np.any(r.valid_moments)
    assert r.fallback_applied.tolist() == [False, False, True]


def _conduct(c):
    from mdanderson_stats import prt_legacy_conduct_decision

    k = c["current"] - 1
    current = prt_legacy_predictive(c["alpha"], c["beta"], c["pending"][k])
    next_risk = (
        prt_legacy_predictive(c["alpha"], c["beta"], c["pending"][k + 1])
        if k + 1 < len(c["pending"])
        else None
    )
    pn = current.negligible if current.pending else 0.0
    pe = current.excessive if current.pending else 0.0
    npe = next_risk.excessive if next_risk is not None and next_risk.pending else 0.0
    result = prt_legacy_conduct_decision(
        c["exceedance"], c["assigned"], c["assessed"], k, pn, pe, npe
    )
    return result


@pytest.mark.parametrize("c", cases("controller"))
def test_full_conduct_code(c):
    result = _conduct(c)
    assert float(result.rule) == pytest.approx(c["code"], abs=5e-7)
    if result.source_dose >= len(c["exceedance"]):
        assert result.action == "legacy_invalid_escalation" and result.warnings
    if result.rule == "6.0":
        assert result.action == "legacy_no_output" and result.warnings


@pytest.mark.parametrize("c", cases("output-dispatch"))
def test_native_output_dispatch_matches_diagnostic_action(c):
    from mdanderson_stats import prt_legacy_conduct_decision

    if c["code"] == 6.33:
        # Abstract late-branch input; no-pending next PE is normally reset to0.
        result = prt_legacy_conduct_decision([0.1, 0.1], [3, 3], [3, 3], 0, 0, 0, 0.96)
    else:
        native = next(v for v in cases("controller") if abs(v["code"] - c["code"]) < 5e-7)
        result = _conduct(native)
    expected = (
        []
        if result.action == "legacy_no_output"
        else ["escalate" if result.action == "legacy_invalid_escalation" else result.action]
    )
    assert c["decision_branches"] == expected


def test_singular_projection_is_not_regularized():
    with pytest.raises(ArithmeticError, match="singular"):
        prt_legacy_projection(np.full((10, 1, 3), 0.5))


@pytest.mark.parametrize(
    "args", [(0, 3, [0.2]), (1, 0, [0.2]), (float("nan"), 3, [0.2]), (1, 3, [0.2] * 11)]
)
def test_invalid_predictive_inputs(args):
    with pytest.raises(ValueError):
        prt_legacy_predictive(*args)


@pytest.mark.parametrize(
    "mean,sd", [([0.5], [-0.1]), ([0.5], [float("nan")]), ([], []), ([0.5, 0.5], [0.1])]
)
def test_invalid_moment_inputs(mean, sd):
    with pytest.raises(ValueError):
        prt_legacy_beta_moments(mean, sd)


def test_controller_dimension_and_history_constraints():
    from mdanderson_stats import prt_legacy_conduct_decision

    with pytest.raises(ValueError):
        prt_legacy_conduct_decision([0.1, 0.2], [3, 3], [4, 3], 0, 0.5, 0.1, 0.1)
    with pytest.raises(ValueError):
        prt_legacy_simulation_decision([0.2, 0.1], [0.1, 0.2], 0, 0, 0.5, 0.1, 0.1)
    with pytest.raises(ValueError):
        prt_legacy_final_selection([3, 3], [0.1, 0.2], [0.1], 0)
    with pytest.raises(ValueError):
        prt_legacy_screening([3, 3], 3, 3, 1, 0, final=1)
