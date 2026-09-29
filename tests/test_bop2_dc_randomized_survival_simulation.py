from copy import deepcopy

import numpy as np
import pytest

from mdanderson_stats.bop2_dc_randomized_survival import bop2_dc_randomized_survival_design
from mdanderson_stats.bop2_dc_randomized_survival_simulation import (
    simulate_bop2_dc_randomized_survival,
)


def _design(*, max_subjects=4, looks=(2, 4), graduate_at_interim=False, priors=None):
    control_prior, treatment_prior = priors or ((2.0, 1.0), (2.0, 1.0))
    return bop2_dc_randomized_survival_design(
        max_subjects=max_subjects,
        median_lrv=0.0,
        median_cmv=0.5,
        control_prior=control_prior,
        treatment_prior=treatment_prior,
        arm_assignments=[0, 1] * (max_subjects // 2),
        looks=looks,
        lambda_lrv=0.1,
        lambda_cmv=0.1,
        graduate_at_interim=graduate_at_interim,
    )


def test_fixed_allocation_simulation_replays_and_conserves_arm_counts():
    design = _design()
    first = simulate_bop2_dc_randomized_survival(
        design,
        2.0,
        3.0,
        accrual_rate=4.0,
        final_followup=0.5,
        n_trials=6,
        rng=419,
    )
    replay = simulate_bop2_dc_randomized_survival(
        design,
        2.0,
        3.0,
        accrual_rate=4.0,
        final_followup=0.5,
        n_trials=6,
        rng=first.rng_seed,
    )
    np.testing.assert_array_equal(first.decision, replay.decision)
    np.testing.assert_array_equal(first.arm_events, replay.arm_events)
    np.testing.assert_array_equal(first.arm_exposure, replay.arm_exposure)
    np.testing.assert_array_equal(first.arm_n.sum(axis=1), first.enrolled)
    np.testing.assert_array_equal(first.decision_count.sum(), first.trials)
    np.testing.assert_allclose(first.decision_probability.sum(), 1.0)
    assert not first.arm_exposure.flags.writeable


def test_time_rescaling_preserves_paths_and_decisions():
    design = _design()
    base = simulate_bop2_dc_randomized_survival(
        design,
        2.0,
        3.0,
        accrual_rate=4.0,
        final_followup=0.5,
        n_trials=4,
        rng=82,
    )
    factor = 1e40
    scaled_design = bop2_dc_randomized_survival_design(
        max_subjects=4,
        median_lrv=0.0,
        median_cmv=0.5 * factor,
        control_prior=(2.0, 1.0 * factor),
        treatment_prior=(2.0, 1.0 * factor),
        arm_assignments=[0, 1, 0, 1],
        looks=[2, 4],
        lambda_lrv=0.1,
        lambda_cmv=0.1,
    )
    scaled = simulate_bop2_dc_randomized_survival(
        scaled_design,
        2.0 * factor,
        3.0 * factor,
        accrual_rate=4.0 / factor,
        final_followup=0.5 * factor,
        n_trials=4,
        rng=82,
    )
    np.testing.assert_array_equal(scaled.decision, base.decision)
    np.testing.assert_array_equal(scaled.arm_events, base.arm_events)
    np.testing.assert_allclose(scaled.duration, factor * base.duration, rtol=2e-14)
    np.testing.assert_allclose(scaled.arm_exposure, factor * base.arm_exposure, rtol=2e-14)


def test_quadrature_preflight_does_not_consume_generator_state():
    n = 1000
    design = _design(max_subjects=n, looks=tuple(range(1, n + 1)))
    rng = np.random.default_rng(51)
    state_before = deepcopy(rng.bit_generator.state)
    with pytest.raises(ValueError, match="total inverse-gamma quadrature work"):
        simulate_bop2_dc_randomized_survival(
            design,
            2.0,
            3.0,
            accrual_rate=4.0,
            final_followup=0.5,
            n_trials=8,
            rng=rng,
        )
    assert rng.bit_generator.state == state_before


def test_early_graduations_are_counted_as_terminal_decisions():
    design = _design(
        graduate_at_interim=True,
        priors=((2.0, 0.05), (2.0, 100.0)),
    )
    result = simulate_bop2_dc_randomized_survival(
        design,
        1.0,
        100.0,
        accrual_rate=4.0,
        final_followup=1.0,
        n_trials=4,
        rng=947,
    )
    assert np.count_nonzero(result.decision == "graduate") > 0
    assert result.decision_count.sum() == result.trials
    np.testing.assert_allclose(result.decision_probability.sum(), 1.0)
