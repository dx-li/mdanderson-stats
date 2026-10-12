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
