import numpy as np
import pytest

from mdanderson_stats.bop2_dc_randomized_survival import (
    BOP2DCRandomizedSurvivalDesign,
    bop2_dc_randomized_survival_design,
    run_bop2_dc_randomized_survival_trial,
)


def _design(**kwargs):
    defaults = dict(
        max_subjects=2,
        median_lrv=0.0,
        median_cmv=1.0,
        control_prior=(1.0, 1.0),
        treatment_prior=(1.0, 1.0),
        arm_assignments=[0, 1],
        looks=[2],
        lambda_lrv=0.1,
        lambda_cmv=0.1,
    )
    defaults.update(kwargs)
    return bop2_dc_randomized_survival_design(**defaults)


def test_zero_margin_inverse_gamma_ordering_and_equal_tail_symmetry():
    design = _design()
    state = design.monitor(0, 0.0, 1, 0, 0.0, 1)
    assert state.posterior_lrv.item() == 0.5
    assert state.absolute_error_lrv.item() == 0.0


def test_zero_duration_event_updates_shape_without_exposure():
    design = _design()
    state = design.monitor(1, 0.0, 1, 0, 0.0, 1)
    np.testing.assert_array_equal(state.posterior_shape, [2.0, 1.0])
    np.testing.assert_array_equal(state.posterior_scale, [1.0, 1.0])


def test_nonzero_median_margin_probability_is_time_unit_invariant():
    base = _design(
        median_lrv=0.2,
        median_cmv=0.4,
        control_prior=(2.0, 0.7),
        treatment_prior=(3.0, 1.2),
    )
    state = base.monitor(0, 0.5, 1, 0, 0.4, 1)
    factor = 1e100
    rescaled = _design(
        median_lrv=0.2 * factor,
        median_cmv=0.4 * factor,
        control_prior=(2.0, 0.7 * factor),
        treatment_prior=(3.0, 1.2 * factor),
    ).monitor(0, 0.5 * factor, 1, 0, 0.4 * factor, 1)
    np.testing.assert_allclose(rescaled.posterior_lrv, state.posterior_lrv, rtol=2e-9)
    np.testing.assert_allclose(rescaled.posterior_cmv, state.posterior_cmv, rtol=2e-9)


def test_calendar_replay_uses_as_of_exposure_by_fixed_arm():
    design = _design(
        max_subjects=4,
        median_lrv=-10.0,
        median_cmv=10.0,
        arm_assignments=[0, 1, 0, 1],
        looks=[2, 4],
        lambda_lrv=0.01,
        lambda_cmv=0.01,
    )
    replay = run_bop2_dc_randomized_survival_trial(
        design,
        [0.0, 1.0, 2.0, 3.0],
        [np.inf, np.inf, 0.5, np.inf],
        final_followup=1.0,
    )
    assert replay.enrolled == 4
    assert replay.decision in {"final_go", "final_consider", "final_no_go"}
    np.testing.assert_allclose(replay.calendar_times, [1.0, 4.0])
    first, final = replay.states
    assert (first.control_events.item(), first.treatment_events.item()) == (0, 0)
    np.testing.assert_allclose(
        [first.control_exposure.item(), first.treatment_exposure.item()], [1, 0]
    )
    assert (final.control_events.item(), final.treatment_events.item()) == (1, 0)
    np.testing.assert_allclose(
        [final.control_exposure.item(), final.treatment_exposure.item()], [4.5, 4.0]
    )


def test_replay_quadrature_budget_rejects_before_monitor(monkeypatch):
    design = _design(
        max_subjects=1000,
        median_lrv=0.1,
        median_cmv=0.2,
        arm_assignments=[0, 1] * 500,
        looks=np.arange(1, 1001),
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("posterior monitor ran before replay preflight")

    monkeypatch.setattr(BOP2DCRandomizedSurvivalDesign, "monitor", fail_if_called)
    with pytest.raises(ValueError, match="total inverse-gamma quadrature work"):
        run_bop2_dc_randomized_survival_trial(
            design,
            np.arange(1000, dtype=float),
            np.full(1000, np.inf),
            final_followup=1.0,
        )
