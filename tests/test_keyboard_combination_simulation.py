"""Cohort simulation tests independent of KeyboardComb posterior arithmetic."""

from dataclasses import dataclass

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from mdanderson_stats.keyboard_combination_simulation import (
    simulate_keyboard_combination,
)


@dataclass(frozen=True)
class _Decision:
    action: str
    next_dose: tuple[int, int] | None
    eliminated: np.ndarray


@dataclass(frozen=True)
class _Selection:
    dose: tuple[int, int] | None


class _PathDesign:
    """Small deterministic core double for testing simulation orchestration."""

    def next_dose(self, patients, toxicities, current_dose, *, eliminated=None, rng=None):
        del rng
        del toxicities
        state = np.zeros_like(patients, dtype=bool) if eliminated is None else eliminated.copy()
        if patients[current_dose[0] - 1, current_dose[1] - 1] >= 6:
            return _Decision("stop_precision", None, state)
        if current_dose == (1, 1):
            return _Decision("escalate", (1, 2), state)
        return _Decision("stay", current_dose, state)

    def select_mtd(self, patients, toxicities, *, eliminated=None):
        del patients, toxicities, eliminated
        return _Selection((1, 2))


class _SafetyDesign(_PathDesign):
    def next_dose(self, patients, toxicities, current_dose, *, eliminated=None, rng=None):
        del rng
        state = np.zeros_like(patients, dtype=bool) if eliminated is None else eliminated.copy()
        if current_dose == (1, 1) and toxicities[0, 0] > 0:
            state[:, :] = True
            return _Decision("stop_safety", None, state)
        return super().next_dose(patients, toxicities, current_dose, eliminated=state)

    def select_mtd(self, patients, toxicities, *, eliminated=None):
        del patients, toxicities, eliminated
        return _Selection(None)


def test_cohorts_are_observed_before_each_core_transition():
    result = simulate_keyboard_combination(
        _PathDesign(),
        [[0, 0, 0], [0, 0, 0]],
        cohorts=3,
        cohort_size=3,
        trials=2,
        rng=123,
    )
    assert_array_equal(result.patients[:, 0, :], [[3, 6, 0], [3, 6, 0]])
    assert_array_equal(result.toxicities, 0)
    assert_array_equal(result.selected_dose, [[1, 2], [1, 2]])
    assert result.stop_reason == ("stop_precision", "stop_precision")
    assert not result.patients.flags.writeable


def test_safety_stopping_and_no_selection_are_recorded():
    result = simulate_keyboard_combination(
        _SafetyDesign(),
        [[1, 0], [0, 0]],
        cohorts=4,
        cohort_size=3,
        trials=4,
        rng=5,
    )
    assert_array_equal(result.patients[:, 0, 0], 3)
    assert_array_equal(result.toxicities[:, 0, 0], 3)
    assert result.stop_reason == ("stop_safety",) * 4
    assert_array_equal(result.selected_dose, 0)
    assert result.selection_probability[0, 0] == 1
    assert result.selection_mcse[0, 0] == 0


def test_seeded_replicates_have_binomial_mcse_and_reproducible_paths():
    kwargs = dict(
        cohorts=2,
        cohort_size=3,
        trials=40,
        rng=42,
    )
    first = simulate_keyboard_combination(_PathDesign(), np.full((2, 2), 0.4), **kwargs)
    second = simulate_keyboard_combination(_PathDesign(), np.full((2, 2), 0.4), **kwargs)
    assert_array_equal(first.patients, second.patients)
    assert_array_equal(first.toxicities, second.toxicities)
    assert_array_equal(first.selected_dose, second.selected_dose)
    expected = np.sqrt(
        first.selection_probability * (1 - first.selection_probability) / kwargs["trials"]
    )
    np.testing.assert_allclose(first.selection_mcse, expected)


@pytest.mark.parametrize("bad", [np.zeros((2, 1)), np.zeros((1, 2)), np.full((2, 2), -0.1)])
def test_truth_grid_validation(bad):
    with pytest.raises(ValueError):
        simulate_keyboard_combination(_PathDesign(), bad, trials=2)
