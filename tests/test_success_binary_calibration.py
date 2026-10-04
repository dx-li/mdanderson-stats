import numpy as np
import pytest

from mdanderson_stats import SuccessOperatingCharacteristics
from mdanderson_stats.success_calibration import (
    binary_success_oc,
    calibrate_success_cutoff,
)
from mdanderson_stats.success_calibration_binary_search import (
    calibrate_binary_success_cutoff,
)


def test_exact_breakpoint_uses_strict_posterior_rule() -> None:
    result = calibrate_binary_success_cutoff(
        1,
        0.3,
        cutoff_range=(0.2, 0.8),
        margin=0.5,
        design_prior=(1, 1),
        analysis_prior=(1, 1),
    )

    assert result.cutoff == pytest.approx(0.25)
    assert result.candidates_evaluated == 2
    assert result.operating_characteristics == binary_success_oc(1, 0.25, margin=0.5)
    assert result.operating_characteristics.false_positive == pytest.approx(0.125)
    assert result.operating_characteristics.true_positive == pytest.approx(0.375)


@pytest.mark.parametrize(
    ("direction", "margin", "design_prior", "analysis_prior", "bounds"),
    [
        ("greater", 0.3, (2, 5), (1, 1), (0.4, 0.99)),
        ("less", 0.6, (1, 3), (2, 1), (0.05, 0.8)),
        ("greater", 0.45, (0.7, 2.4), (3.2, 1.1), (0, 1)),
    ],
)
def test_search_matches_exhaustive_breakpoint_evaluation(
    direction: str,
    margin: float,
    design_prior: tuple[float, float],
    analysis_prior: tuple[float, float],
    bounds: tuple[float, float],
) -> None:
    n = 7
    lower, upper = bounds
    # Recover the finite decision breakpoints independently from the beta
    # posterior formula, then use the existing caller-grid calibrator as oracle.
    from scipy.special import betainc, betaincc

    counts = np.arange(n + 1)
    tail = betaincc if direction == "greater" else betainc
    q = tail(analysis_prior[0] + counts, analysis_prior[1] + n - counts, margin)
    candidates = np.unique(np.concatenate(([lower], q[(q > lower) & (q <= upper)])))

    def evaluate(cutoff: float) -> SuccessOperatingCharacteristics:
        return binary_success_oc(
            n,
            cutoff,
            margin=margin,
            design_prior=design_prior,
            analysis_prior=analysis_prior,
            direction=direction,
        )

    expected = calibrate_success_cutoff(evaluate, 0.2, candidates)
    actual = calibrate_binary_success_cutoff(
        n,
        0.2,
        cutoff_range=bounds,
        margin=margin,
        design_prior=design_prior,
        analysis_prior=analysis_prior,
        direction=direction,
    )
    assert actual.cutoff == expected.cutoff
    assert 1 <= actual.candidates_evaluated <= 1 + (len(candidates) - 1).bit_length()
    assert actual.operating_characteristics == expected.operating_characteristics


def test_single_point_range_and_no_feasible_cutoff() -> None:
    one = calibrate_binary_success_cutoff(1, 0.6, cutoff_range=(0.2, 0.2), margin=0.5)
    assert one.cutoff == 0.2
    with pytest.raises(ValueError, match="no cutoff"):
        calibrate_binary_success_cutoff(1, 0.1, cutoff_range=(0.2, 0.8), margin=0.5)
