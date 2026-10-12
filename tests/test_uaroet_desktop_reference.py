"""Original UAROET1.8 numerical references, distinct from controller validation."""

import json
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import uaroet_logits, uaroet_probabilities

REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/uaroet-desktop-marginals.json").read_text()
)


def native_coordinates(case, levels):
    """Explicitly transform native stride-five values, including its increment rule."""
    native = np.array(case["parameters"][:30]).reshape(2, 3, 5)
    parts = []
    for endpoint in native:
        active = endpoint[: levels - 1, : case["doses"]].copy()
        active[:, 1:] = (
            np.exp(active[:, 1:]) if case["lognormal"] else np.maximum(active[:, 1:], 1e-10)
        )
        parts.extend(active[:, 0])
        parts.extend(active[:, 1:].T.ravel())
    return parts


@pytest.mark.parametrize("case", REFERENCE["eta"])
def test_native_eta(case):
    logits = uaroet_logits(
        native_coordinates(case, 4), dose_count=case["doses"], efficacy_levels=4, toxicity_levels=4
    )
    assert logits[case["endpoint"]][case["dose"], case["category"] - 1] == pytest.approx(
        case["value"], abs=2e-14
    )


@pytest.mark.parametrize("case", REFERENCE["marginals"])
def test_native_continuation_marginals(case):
    levels = case["categories"]
    logits = uaroet_logits(
        native_coordinates(case, levels),
        dose_count=case["doses"],
        efficacy_levels=levels,
        toxicity_levels=levels,
    )
    result = uaroet_probabilities(*logits)
    actual = (result.efficacy, result.toxicity)[case["endpoint"]][case["dose"]]
    np.testing.assert_allclose(actual, case["value"][:levels], atol=2e-14, rtol=2e-13)
    assert sum(case["value"][:levels]) == pytest.approx(1.0)


SCREEN = json.loads(
    (Path(__file__).parent / "fixtures/uaroet-desktop-acceptability.json").read_text()
)


@pytest.mark.parametrize("case", SCREEN["cases"])
def test_native_screening_contract(case):
    x = case["inputs"]
    tails = x["tails"]
    n = len(tails)
    threshold = x.get("threshold", 0.8)
    efficacy = x.get("eff_tails", [0] * n)
    utility = x.get("utility", [20, 40, 60])
    best = x.get("best", [0.1, 0.2, 0.7])
    statuses = []
    prior_toxic = False
    for i in range(n):
        if i > x.get("highest", 2):
            status = -5
        elif tails[i] > threshold and (x["bad_tox"][i] > 0 or prior_toxic):
            status = -2
            prior_toxic = True
        elif efficacy[i] > threshold and x["bad_eff"][i] >= x.get("min_bad_eff", 1):
            status = -1
        else:
            status = 1
        statuses.append(status)
    for i in range(n):
        if (
            statuses[i] != 1
            or x.get("observed", [3] * n)[i] < x.get("min_observed", 0)
            or not x.get("filter_utility", True)
        ):
            continue
        if max(utility) - utility[i] > x.get("delta", 100):
            statuses[i] = -3
        elif best[i] < x.get("min_best", 0):
            statuses[i] = -4
    if 1 not in statuses and statuses[0] == -2 and sum(x["bad_tox"]) < x.get("min_total_tox", 0):
        statuses[0] = 1
    assert statuses == case["status"]


RANDOMIZATION = json.loads(
    (Path(__file__).parent / "fixtures/uaroet-desktop-randomization.json").read_text()
)


@pytest.mark.parametrize("case", RANDOMIZATION["cases"])
def test_native_hybrid_randomization_contract(case):
    x = case["inputs"]
    eligible = [i for i, status in enumerate(x["status"]) if status == 1]
    best = max(eligible, key=lambda i: (x["utility"][i], i))
    weights = np.zeros(len(x["status"]))
    allocations = sorted((x["treated"][i] for i in eligible), reverse=True)
    greedy = len(eligible) == 1 or (
        x["hybrid"]
        and (
            x["treated"][best] != allocations[0]
            or x["treated"][best] - allocations[1] < x["surplus"]
        )
    )
    if greedy:
        weights[best] = 1.0
        selected = best
    else:
        raw = x["utility"] if x["mean_weight"] else [0.2, 0.4, 0.6]
        weights[eligible] = np.array(raw)[eligible]
        weights /= weights.sum()
        selected = next(i for i in eligible if x["value"] <= weights.cumsum()[i])
    np.testing.assert_allclose(case["result"]["weights"], weights, atol=2e-15)
    assert case["result"]["selected"] == selected
