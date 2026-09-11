from pathlib import Path

import numpy as np

from mdanderson_stats.beta_binomial import BetaBinomialPosterior
from mdanderson_stats.beta_comparison import compare_beta_difference
from mdanderson_stats.success_calibration import (
    binary_two_arm_success_oc,
    calibrate_success_cutoff,
    prepare_binary_two_arm_success,
)


def test_shifted_beta_against_independent_r():
    rows = np.genfromtxt(
        Path(__file__).parent / "fixtures/beta-difference-r.csv", delimiter=",", names=True
    )
    for row in rows:
        result = compare_beta_difference(
            BetaBinomialPosterior(row["a"], row["b"]),
            BetaBinomialPosterior(row["c"], row["d"]),
            row["margin"],
            absolute_tolerance=1e-10,
        )
        np.testing.assert_allclose(
            [result.above_margin, result.below_margin],
            [row["above"], row["below"]],
            atol=2e-10,
            rtol=0,
        )
    prior = BetaBinomialPosterior()
    for delta in (-1.0, -0.5, 0.0, 0.5, 1.0):
        r = compare_beta_difference(prior, prior, delta)
        expected = (1 - delta) ** 2 / 2 if delta >= 0 else 1 - (1 + delta) ** 2 / 2
        np.testing.assert_allclose(r.above_margin, expected, atol=1e-9)


def test_reusable_margin_table():
    args = dict(
        design_treatment=(3.0, 7.0),
        design_control=(2.0, 8.0),
        margin=0.15,
        null_rate=0.2,
        null_treatment_rate=0.35,
    )
    table = prepare_binary_two_arm_success(5, 4, **args)
    for c in (0.7, 0.9, 0.95):
        assert table.evaluate(c) == binary_two_arm_success_oc(5, 4, c, **args)
    chosen = calibrate_success_cutoff(table.evaluate, 0.1, np.linspace(0.5, 0.999, 30))
    assert chosen.operating_characteristics.incorrect_decision_probability <= 0.1
    # Direction reversal plus exchanged arms preserves the original decision.
    reflected = prepare_binary_two_arm_success(
        4,
        5,
        design_treatment=(2.0, 8.0),
        design_control=(3.0, 7.0),
        margin=-0.15,
        null_rate=0.35,
        null_treatment_rate=0.2,
        direction="less",
    )
    np.testing.assert_allclose(
        table.posterior_probability, reflected.posterior_probability.T, atol=1e-12
    )
    assert not table.posterior_probability.flags.writeable
    impossible = prepare_binary_two_arm_success(2, 2, margin=1.0)
    assert impossible.evaluate(0).bayesian_power == 0
    certain = prepare_binary_two_arm_success(2, 2, margin=-1.0)
    assert abs(certain.evaluate(0).bayesian_power - 1) < 1e-12
