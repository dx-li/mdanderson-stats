import numpy as np
import pytest
from scipy.special import logit

from mdanderson_stats.efftox_decision import efftox_decision
from mdanderson_stats.efftox_trinary_contour import EffToxTrinaryContour
from mdanderson_stats.efftox_trinary_model import EffToxTrinaryPrior, fit_efftox_trinary


def test_report_contour_shape_intercept_target_utility_and_half_ray():
    contour = EffToxTrinaryContour.from_points([0.45, 0.55, 0.84], [0.0, 0.10, 0.16])
    assert contour.shape == pytest.approx(2.10360951613587, rel=0, abs=2e-13)
    assert contour.toxicity_intercept == pytest.approx(0.165995389836415, rel=0, abs=2e-13)
    np.testing.assert_allclose(
        contour.utility([0.45, 0.55, 0.84], [0.0, 0.10, 0.16]), 0.0, atol=2e-14
    )
    assert contour.utility(1.0, 0.0) == pytest.approx(1.0)
    assert contour.utility(0.775, 0.05) == pytest.approx(0.5, rel=0, abs=2e-14)
    assert contour.utility(0.2, 0.8) < 0.0


def test_contour_supports_known_shapes_and_rejects_misplaced_hypotenuse_point():
    for shape, toxicity_intercept in ((1.0, 0.3), (2.0, 0.4), (0.5, 2.0)):
        e0 = 0.4
        remainder = 1.0 - e0
        toxicity_high = (remainder ** (-shape) + toxicity_intercept ** (-shape)) ** (-1.0 / shape)
        efficacy_high = 1.0 - toxicity_high
        efficacy_middle = (e0 + efficacy_high) / 2.0
        toxicity_middle = toxicity_intercept * (
            1.0 - ((1.0 - efficacy_middle) / remainder) ** shape
        ) ** (1.0 / shape)
        contour = EffToxTrinaryContour.from_points(
            [e0, efficacy_middle, efficacy_high],
            [0.0, toxicity_middle, toxicity_high],
        )
        assert contour.shape == pytest.approx(shape, rel=0, abs=3e-12)
        assert contour.toxicity_intercept == pytest.approx(toxicity_intercept, rel=0, abs=3e-12)

    with pytest.raises(ValueError, match=r"eh\+th=1"):
        EffToxTrinaryContour.from_points([0.2, 0.4, np.nextafter(1.0, 0.0)], [0.0, 1e-301, 1e-300])

    close_low_points = EffToxTrinaryContour.from_points([1e-16, 2e-16, 0.9], [0.0, 0.05, 0.1])
    assert np.isfinite(close_low_points.shape)
    np.testing.assert_allclose(
        close_low_points.utility([1e-16, 2e-16, 0.9], [0.0, 0.05, 0.1]),
        0.0,
        atol=2e-14,
    )


def test_decision_uses_marginal_not_conditional_efficacy_probability():
    prior = EffToxTrinaryPrior(
        mean=[0.0, 0.1, logit(0.6), 0.01],
        sd=[0.0, 0.0, 0.0, 0.0],
    )
    counts = np.asarray([[0, 1, 0], [0, 0, 0]])
    fit = fit_efftox_trinary(
        [1.0, 2.0],
        counts,
        prior=prior,
        draws=8,
        warmup=0,
        chains=2,
        rng=np.random.default_rng(4),
    )
    contour = EffToxTrinaryContour.from_points([0.2, 0.5, 0.8], [0.0, 0.1, 0.2])
    decision = efftox_decision(
        fit,
        contour,
        efficacy_limit=0.4,
        toxicity_limit=0.9,
        efficacy_probability=0.5,
        toxicity_probability=0.5,
        starting_dose=1,
        phase="final",
        allow_untried_exploration=False,
    )
    assert np.all(fit.conditional_efficacy_probabilities > 0.59)
    assert np.all(fit.efficacy_probabilities < 0.4)
    assert np.all(decision.efficacy_tail_probability == 0.0)
    assert decision.action == "stop_no_admissible"


def test_trinary_interim_keeps_no_lookskip_while_final_can_select_high_dose():
    prior = EffToxTrinaryPrior(
        mean=[-3.0, 0.1, 0.0, 2.0],
        sd=[0.0, 0.0, 0.0, 0.0],
    )
    counts = np.asarray([[1, 0, 0], [0, 0, 0], [1, 0, 0]])
    fit = fit_efftox_trinary(
        [1.0, 2.0, 4.0],
        counts,
        prior=prior,
        draws=8,
        warmup=0,
        chains=2,
        rng=np.random.default_rng(8),
    )
    contour = EffToxTrinaryContour.from_points([0.1, 0.6, 0.9], [0.0, 0.05, 0.1])
    common = dict(
        fit=fit,
        contour=contour,
        efficacy_limit=0.0,
        toxicity_limit=1.0,
        efficacy_probability=0.0,
        toxicity_probability=0.0,
        starting_dose=1,
        allow_untried_exploration=True,
    )
    interim = efftox_decision(last_dose=1, phase="interim", **common)
    final = efftox_decision(phase="final", **common)
    assert interim.action == "assign"
    assert interim.dose in (1, 2)
    assert final.action == "select"
    assert final.dose == 3
