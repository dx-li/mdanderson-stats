import csv
from pathlib import Path

import numpy as np

from mdanderson_stats.top_endpoints import TOPMultiEndpointDesign

FIXTURES = Path(__file__).parent / "fixtures"


def _design(
    null: float, *, endpoint: int, toxicity: bool, maximum: int, scale: float, gamma: float
):
    rates = [null, 0.3]
    if endpoint == 1:
        rates = [0.15, null]
    p1, p2 = rates
    joint = [p1 * p2, p1 * (1 - p2), (1 - p1) * p2, (1 - p1) * (1 - p2)]
    return TOPMultiEndpointDesign(
        maximum,
        joint,
        scale,
        gamma,
        mode="efficacy_toxicity" if toxicity else "coprimary",
    )


def _settings():
    with (FIXTURES / "top-endpoints-settings.csv").open(newline="") as stream:
        return {row["case"]: row for row in csv.DictReader(stream)}


def test_endpoint_marginals_match_base_r_posterior_and_boundary_references():
    settings = _settings()
    with (FIXTURES / "top-endpoints-posterior.csv").open(newline="") as stream:
        posterior = list(csv.DictReader(stream))
    designs = {}
    for name, cfg in settings.items():
        endpoint = int(cfg["direction"] == "toxicity")
        designs[name] = _design(
            float(cfg["null"]),
            endpoint=endpoint,
            toxicity=endpoint == 1,
            maximum=int(cfg["maximum"]),
            scale=float(cfg["scale"]),
            gamma=float(cfg["gamma"]),
        )
    boundary_tables = {name: design.boundaries() for name, design in designs.items()}
    for row in posterior:
        design = designs[row["case"]]
        j = int(settings[row["case"]]["direction"] == "toxicity")
        n, events, ess = int(row["patients"]), int(row["events"]), float(row["effective_size"])
        pending = n - events
        endpoint_events = [0, 0]
        endpoint_pending = [0, 0]
        endpoint_weight = [0.0, 0.0]
        endpoint_events[j] = events
        endpoint_pending[j] = pending
        endpoint_weight[j] = ess - events
        result = design.evaluate(n, endpoint_events, endpoint_pending, endpoint_weight)
        np.testing.assert_allclose(result.posterior_alpha[j], float(row["shape1"]), atol=2e-14)
        np.testing.assert_allclose(result.posterior_beta[j], float(row["shape2"]), atol=2e-14)
        np.testing.assert_allclose(
            result.acceptable_probability[j], float(row["acceptable_probability"]), atol=2e-13
        )
        np.testing.assert_allclose(result.cutoff[j], float(row["cutoff"]), atol=2e-14)
        assert (result.acceptable_probability[j] < result.cutoff[j]) == (
            row["stop"].lower() == "true"
        )

    with (FIXTURES / "top-endpoints-boundaries.csv").open(newline="") as stream:
        boundary_rows = list(csv.DictReader(stream))
    for row in boundary_rows:
        design = designs[row["case"]]
        j = int(settings[row["case"]]["direction"] == "toxicity")
        boundary = boundary_tables[row["case"]]
        look_index = int(np.flatnonzero(boundary.patients == int(row["patients"]))[0])
        r = int(row["events"])
        expected = row["effective_size_crossing"]
        actual = boundary.effective_size_crossing[look_index, j, r]
        if expected:
            np.testing.assert_allclose(actual, float(expected), atol=2e-9)
        else:
            assert np.isnan(actual)


def test_source_timing_mixture_and_patient_level_partial_statuses():
    design = TOPMultiEndpointDesign(
        6,
        [0.2, 0.3, 0.1, 0.4],
        0.8,
        0.5,
        windows=[3, 6],
        timing_probabilities=[[0.6, 0.3, 0.1], [0.2, 0.3, 0.5]],
        looks=[2, 4, 6],
    )
    weights = design.timing_weight([[1, 2], [2, 4], [3, 6]])
    np.testing.assert_allclose(weights[:, 0], [0.6, 0.9, 1])
    np.testing.assert_allclose(weights[:, 1], [0.2, 0.5, 1])
    outcomes = np.array([[1, np.nan], [0, 1], [np.nan, 0], [1, 1]])
    followup = np.array([[0, 2], [0, 0], [2, 0], [0, 0]], dtype=float)
    by_patient = design.evaluate_followup(outcomes, followup)
    summary = design.evaluate(4, [2, 2], [1, 1], [0.9, 0.2])
    np.testing.assert_allclose(by_patient.effective_sample_size, summary.effective_sample_size)


def test_mode_combination_and_endpoint_specific_suspension():
    joint = [0.25, 0.25, 0.25, 0.25]
    co = TOPMultiEndpointDesign(4, joint, 0.5, 0, looks=[2, 4])
    # One co-primary endpoint is acceptable even when the other remains pending.
    res = co.evaluate(2, [2, 0], [0, 2], [0, 0])
    assert res.endpoint_status.tolist() == ["acceptable", "suspend"]
    assert res.decision.tolist() == "continue"
    final = co.evaluate(4, [4, 0], [0, 4], [0, 0])
    assert final.endpoint_status.tolist() == ["acceptable", "suspend"]
    assert final.decision.tolist() == "success"

    et = TOPMultiEndpointDesign(4, joint, 0.5, 0, mode="efficacy_toxicity", looks=[2, 4])
    tox = et.evaluate(2, [2, 2], [0, 0], [0, 0])
    assert tox.decision.tolist() == "stop_toxicity"
    both = et.evaluate(2, [0, 2], [0, 0], [0, 0])
    assert both.decision.tolist() == "stop_futility_toxicity"
    waiting = et.evaluate(2, [0, 0], [2, 2], [0, 0])
    assert waiting.endpoint_status.tolist() == ["suspend", "suspend"]
    assert waiting.decision.tolist() == "suspend"


def test_batching_tiny_prior_and_input_arrays_remain_mutable():
    windows = np.array([2.0, 3.0])
    timing = np.array([[0.2, 0.3, 0.5], [0.4, 0.4, 0.2]])
    design = TOPMultiEndpointDesign(
        4,
        [0.25] * 4,
        0.8,
        0.5,
        prior_concentration=1e-20,
        windows=windows,
        timing_probabilities=timing,
        looks=[2, 4],
    )
    assert windows.flags.writeable and timing.flags.writeable
    tiny_posterior = design.evaluate(2, [2, 0], [0, 0], [0, 0])
    assert tiny_posterior.posterior_beta[0] > 0

    batch = design.evaluate([2, 4], [[2, 0], [3, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]])
    assert batch.posterior_alpha.shape == (2, 2)
    assert batch.decision.shape == (2,)
