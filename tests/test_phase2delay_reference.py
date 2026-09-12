"""Deterministic integrals independently check the stochastic MI posterior."""

import csv
from pathlib import Path

import numpy as np

from mdanderson_stats.phase2delay import phase2_delay_monitor


def test_delayed_posterior_against_integrated_correlated_hazards():
    with (Path(__file__).parent / "fixtures/phase2delay-reference.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    for case in (1, 2):
        kwargs = dict(
            event=[1, 0, 0, 0] if case == 1 else [1, 0, 0],
            event_time=[0.2, 0, 0, 0] if case == 1 else [0.25, 0, 0],
            followup=[0.2, 1, 0.25, 0.75] if case == 1 else [0.25, 1, 0.5],
            window=1,
            threshold=0.4,
            intervals=case,
            prior_alpha=0.3,
            prior_beta=0.7,
            hazard_c=2 if case == 1 else [2, 3],
            lambda0=0.4,
            burn_in=500,
            hazard_draws=4000,
            seed=141,
        )
        for endpoint in ("response", "toxicity", "progression"):
            result = phase2_delay_monitor(**kwargs, endpoint=endpoint)
            reference = next(
                row for row in rows if int(row["case"]) == case and row["endpoint"] == endpoint
            )
            # Fixed seeds, with a conservative absolute bound and a Monte Carlo
            # error check; no comparison to another copy of the sampler.
            error = abs(result.posterior_probability - float(reference["probability"]))
            assert error < 0.025
            assert error < 6 * result.posterior_probability_mc_se + 0.002
            means = [float(reference["hazard_mean_1"])]
            if case == 2:
                means.append(float(reference["hazard_mean_2"]))
            np.testing.assert_allclose(result.hazard_trace.mean(axis=0), means, atol=0.04, rtol=0)


def test_event_times_and_hazards_change_units_together():
    results = []
    for scale in (1e-50, 1.0, 1e50):
        results.append(
            phase2_delay_monitor(
                [1, 0, 0],
                np.array([0.25, 0, 0]) * scale,
                np.array([0.25, 1, 0.5]) * scale,
                window=scale,
                threshold=0.4,
                intervals=2,
                prior_alpha=0.3,
                prior_beta=0.7,
                hazard_c=[2, 3],
                lambda0=0.4 / scale,
                burn_in=100,
                hazard_draws=300,
                seed=141,
            )
        )
    for result in results:
        np.testing.assert_allclose(result.posterior_probability, results[1].posterior_probability)


def test_complete_data_preserves_tiny_prior_shapes_and_direct_tails():
    for endpoint, event, alpha, beta, threshold in (
        ("response", 1, 1.0, 1e-20, 0.4),
        ("toxicity", 0, 1e-20, 1.0, 0.6),
    ):
        result = phase2_delay_monitor(
            [event],
            [event],
            [1],
            window=1,
            endpoint=endpoint,
            threshold=threshold,
            prior_alpha=alpha,
            prior_beta=beta,
            hazard_c=0.01,
            lambda0=0.4,
        )
        # Leading small-shape integral for Beta(2,b), mirrored for the upper
        # tail: b * integral_0^.4 x/(1-x) dx. Relative correction is O(b).
        expected = 1e-20 * (-np.log1p(-0.4) - 0.4)
        np.testing.assert_allclose(result.posterior_probability, expected, rtol=1e-13, atol=0)
        assert result.hazard_draws == 0
        assert result.posterior_probability_mc_se == 0
    near_zero = phase2_delay_monitor(
        [0],
        [0],
        [1],
        window=1,
        threshold=1e-20,
        prior_alpha=1,
        prior_beta=1,
        hazard_c=0.01,
        lambda0=0.4,
    )
    np.testing.assert_allclose(near_zero.posterior_probability, 2e-20, rtol=1e-14, atol=0)
