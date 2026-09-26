import numpy as np
import pytest

from mdanderson_stats.efftox_decision import EffToxContour, efftox_decision
from mdanderson_stats.efftox_model import (
    EffToxPrior,
    efftox_log_joint_probabilities,
    efftox_predict,
    efftox_standardize,
    fit_efftox,
)


def test_standardized_log_doses_and_original_zero_dose_rule() -> None:
    logs = np.log([1, 2, 4])
    np.testing.assert_allclose(efftox_standardize([1, 2, 4]), logs - logs.mean())
    shifted = efftox_standardize([0, 2, 4])
    np.testing.assert_allclose(shifted, np.log([2, 4, 6]) - np.log([2, 4, 6]).mean())


def test_joint_probabilities_are_normalized_and_association_changes_cells() -> None:
    x = np.array([-0.4, 0.4])
    theta = np.array([[-1.0, 0.7, 0.1, 0.4, -0.2, 0.0], [-1.0, 0.7, 0.1, 0.4, -0.2, 1.2]])
    cells = efftox_predict(x, theta)
    np.testing.assert_allclose(cells.sum(axis=(-1, -2)), 1.0)
    np.testing.assert_allclose(cells.sum(axis=-1)[0], cells.sum(axis=-1)[1])
    assert not np.allclose(cells[0], cells[1])


def test_joint_log_cells_keep_extreme_cancellation_terms() -> None:
    x = np.array([-0.5, 0.5])
    theta = np.array([1000.0, 0.0, 1000.0, 0.0, 0.0, -1000.0])
    logp = efftox_log_joint_probabilities(x, theta)
    assert logp[0, 0, 0] == pytest.approx(-3000.0 + np.log(4.0), abs=1e-9)
    theta = np.array([1000.0, 0.0, -1000.0, 0.0, 0.0, 1000.0])
    logp = efftox_log_joint_probabilities(x, theta)
    assert logp[0, 1, 0] == pytest.approx(-3000.0 + np.log(4.0), abs=1e-9)


def test_resource_limits_reject_large_views_and_broadcasts() -> None:
    # Broadcast views contain almost no storage; reject before materializing.
    parameters = np.broadcast_to(np.zeros(6, dtype=np.uint8), (100_001, 6))
    with pytest.raises(ValueError, match="200000"):
        efftox_predict([-0.5, 0.5], parameters)
    contour = EffToxContour.from_points(0.3, 0.6, 0.65, 0.25)
    with pytest.raises(ValueError, match="200000"):
        contour.utility(np.zeros((1000, 1)), np.zeros((1, 1000)))
    with pytest.raises(ValueError, match="budget"):
        fit_efftox(
            np.arange(1, 21),
            np.zeros((20, 2, 2)),
            prior=EffToxPrior(np.zeros(6), np.ones(6)),
            draws=2000,
            warmup=10_000,
            chains=4,
            rng=np.random.default_rng(1),
        )


def test_l_p_contour_fits_three_equal_desirability_points() -> None:
    contour = EffToxContour.from_points(0.3, 0.6, 0.65, 0.25)
    points = contour.utility([0.3, 0.65, 1.0], [0.0, 0.25, 0.6])
    np.testing.assert_allclose(points, 0.0, atol=1e-10)
    assert contour.utility(1.0, 0.0) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="shape does not"):
        EffToxContour(0.3, 0.6, 0.65, 0.25, 1.0)


def test_no_data_fit_samples_explicit_truncated_prior_and_reports_diagnostics() -> None:
    prior = EffToxPrior(
        mean=np.array([-1.0, -1.0, -0.5, 0.2, 0.0, 0.1]),
        sd=np.array([0.5, 0.8, 0.5, 0.4, 0.2, 0.3]),
    )
    fit = fit_efftox(
        [1, 2, 4],
        np.zeros((3, 2, 2)),
        prior=prior,
        draws=8,
        warmup=0,
        chains=2,
        rng=np.random.default_rng(12),
    )
    assert np.all(fit.parameters[..., 1] > 0)
    assert not fit.efficacy_probabilities.flags.writeable
    assert not fit.counts.flags.writeable
    decision = efftox_decision(
        fit,
        EffToxContour.from_points(0.3, 0.6, 0.65, 0.25),
        efficacy_limit=0.3,
        toxicity_limit=0.5,
        efficacy_probability=0.1,
        toxicity_probability=0.1,
        starting_dose=2,
        last_dose=None,
    )
    assert decision.action == "start"
    assert decision.dose == 2
    assert np.any(decision.utility != 0)
    assert np.all(np.isfinite(decision.efficacy_tail_probability))


def test_observed_decisions_apply_directional_skip_rules_and_final_selection() -> None:
    doses = [1, 2, 4, 8]
    counts = np.zeros((4, 2, 2))
    counts[2, 1, 0] = 1
    contour = EffToxContour.from_points(0.3, 0.6, 0.65, 0.25)
    prior = EffToxPrior(
        mean=np.array([-2.0, 0.0, 1.0, -0.2, 0.0, 0.0]),
        sd=np.zeros(6),
        monotone_toxicity=False,
    )
    fit = fit_efftox(
        doses, counts, prior=prior, draws=8, warmup=0, chains=2, rng=np.random.default_rng(13)
    )
    settings = dict(
        fit=fit,
        contour=contour,
        efficacy_limit=0.0,
        toxicity_limit=1.0,
        efficacy_probability=0.1,
        toxicity_probability=0.1,
        starting_dose=1,
        last_dose=3,
        allow_untried_exploration=False,
    )
    both = efftox_decision(**settings, skip_policy="both")
    escalation = efftox_decision(**settings, skip_policy="escalation")
    final = efftox_decision(**settings, phase="final", skip_policy="both")
    assert both.dose == 2
    assert escalation.dose == 1
    assert final.action == "select" and final.dose == 1

    only_low = EffToxPrior(
        mean=np.array([-2.0, 0.0, -0.2, -2.0, 0.0, 0.0]),
        sd=np.zeros(6),
        monotone_toxicity=False,
    )
    low_fit = fit_efftox(
        doses,
        counts,
        prior=only_low,
        draws=8,
        warmup=0,
        chains=2,
        rng=np.random.default_rng(14),
    )
    unreachable = efftox_decision(
        **{**settings, "fit": low_fit, "efficacy_limit": 0.8}, skip_policy="both"
    )
    assert unreachable.action == "stop_no_reachable"
