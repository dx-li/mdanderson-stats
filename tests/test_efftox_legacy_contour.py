import numpy as np
import pytest

from mdanderson_stats.efftox_decision import efftox_decision
from mdanderson_stats.efftox_legacy_contour import EffToxLegacyContour
from mdanderson_stats.efftox_model import EffToxPrior, fit_efftox
from mdanderson_stats.efftox_simulation import simulate_efftox


def _pentostatin_contour():
    return EffToxLegacyContour.from_points([0.15, 0.25, 1.0], [0.0, 0.30, 0.60])


def test_inverse_quadratic_interpolates_and_recovers_published_radial_score():
    contour = _pentostatin_contour()
    np.testing.assert_allclose(
        contour.target_toxicity([0.15, 0.25, 1.0]), [0.0, 0.30, 0.60], atol=2e-15
    )
    target_e = np.asarray([0.15, 0.25, 1.0])
    target_t = np.asarray([0.0, 0.30, 0.60])
    transformed_e = 1.0 - (1.0 - target_e) / 2.0
    transformed_t = target_t / 2.0
    np.testing.assert_allclose(contour.utility(transformed_e, transformed_t), 1.0, atol=2e-13)
    assert contour.utility(0.15, 0.0) == pytest.approx(0.0, abs=2e-15)
    assert contour.utility(0.0, 0.0) == pytest.approx(-0.15)
    assert np.isposinf(contour.utility(1.0, 0.0))
    assert contour.utility(1.0, 1.0) == pytest.approx(-0.4)


def test_inverse_linear_and_inverse_square_degeneracies_are_supported():
    inverse_linear = EffToxLegacyContour.from_points([0.1, 0.5, 1.0], [0.0, 0.64, 0.72])
    inverse_square = EffToxLegacyContour.from_points([0.2, 0.6, 1.0], [0.0, 4 / 9, 0.48])

    assert inverse_linear.c == pytest.approx(0.0, abs=2e-15)
    assert inverse_square.b == pytest.approx(0.0, abs=2e-14)
    for contour in (inverse_linear, inverse_square):
        assert contour.utility(0.625, 0.5 * contour.target_toxicity(0.25)) == pytest.approx(
            1.0, abs=2e-12
        )


def test_legacy_contour_connects_to_existing_decision_and_simulation():
    doses = np.asarray([1.0, 2.0])
    prior = EffToxPrior(mean=[-2.0, 0.5, 1.0, 0.5, -0.1, 0.0], sd=np.zeros(6))
    contour = _pentostatin_contour()
    counts = np.zeros((2, 2, 2), dtype=np.int64)
    counts[0, 0, 0] = 1
    fit = fit_efftox(
        doses,
        counts,
        prior=prior,
        draws=8,
        warmup=0,
        chains=2,
        rng=np.random.default_rng(3),
    )
    decision = efftox_decision(
        fit,
        contour,
        efficacy_limit=0.2,
        toxicity_limit=0.3,
        efficacy_probability=0.0,
        toxicity_probability=0.0,
        starting_dose=1,
        phase="final",
        allow_untried_exploration=False,
    )
    assert decision.action in ("select", "stop_no_admissible")
    truth = np.asarray([[[0.55, 0.15], [0.20, 0.10]], [[0.30, 0.20], [0.35, 0.15]]])
    result = simulate_efftox(
        doses,
        truth,
        prior=prior,
        contour=contour,
        efficacy_limit=0.2,
        toxicity_limit=0.3,
        efficacy_probability=0.0,
        toxicity_probability=0.0,
        cohorts=1,
        cohort_size=1,
        trials=2,
        draws=8,
        warmup=0,
        chains=2,
        allow_untried_exploration=False,
        rng=9,
        sampler_rng=10,
    )
    assert result.outcome_counts.sum() == 2
    assert result.selected_dose.shape == (2,)


def test_legacy_contour_rejects_inadmissible_points_and_broadcasts_are_bounded():
    with pytest.raises(ValueError, match="points must satisfy"):
        EffToxLegacyContour.from_points([0.15, 0.25, 0.99], [0.0, 0.3, 0.6])
    contour = _pentostatin_contour()
    assert contour.utility(0.5, 0.2).shape == ()
    with pytest.raises(ValueError, match="broadcast exceeds"):
        contour.utility(np.zeros((500, 1)), np.zeros((1, 500)))


def test_small_positive_toxicity_approaches_zero_toxicity_score():
    contour = _pentostatin_contour()
    zero_toxicity_score = contour.utility(0.5, 0.0)

    scores = [contour.utility(0.5, toxicity) for toxicity in (1e-12, 1e-20, 1e-100)]
    assert all(np.isfinite(score) for score in scores)
    assert abs(scores[-1] - zero_toxicity_score) < 2e-14
    assert abs(scores[-1] - zero_toxicity_score) < abs(scores[0] - zero_toxicity_score)
