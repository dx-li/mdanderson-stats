import numpy as np
import pytest

from mdanderson_stats.boin import BOINDesign
from mdanderson_stats.boin_time_comparison import (
    compare_tite_boin_rolling_six,
    tite_boin_rolling_six_report,
)
from mdanderson_stats.rolling_six import RollingSixDesign


def test_comparison_preserves_distinct_terminal_statuses_and_summaries() -> None:
    design = BOINDesign(target=0.25)
    unsafe = compare_tite_boin_rolling_six(
        design,
        [1.0, 1.0],
        1.0,
        2.0,
        cohorts=2,
        cohort_size=3,
        rolling_six_max_patients=6,
        trials=3,
        rng=41,
    )
    assert set(unsafe.tite_boin.stop_reason) <= {"max_patients", "stop_safety"}
    assert np.all(unsafe.tite_boin.selected_dose == 0)
    assert set(unsafe.rolling_six.selection_status) == {"no_safe_dose"}
    np.testing.assert_allclose(unsafe.tite_boin_summary.mean_patients_by_dose, [4, 0])
    np.testing.assert_allclose(
        unsafe.rolling_six_summary.mean_patients_by_dose, unsafe.rolling_six.patients.mean(axis=0)
    )
    assert unsafe.tite_boin_summary.status_probabilities == (("no_recommendation", 1.0),)

    safe = compare_tite_boin_rolling_six(
        design,
        [0.0, 0.0],
        1.0,
        2.0,
        cohorts=2,
        cohort_size=3,
        rolling_six_max_patients=12,
        trials=3,
        rolling_six_design=RollingSixDesign(require_complete_before_escalation=True),
        rng=41,
    )
    assert set(safe.rolling_six.selection_status) == {"highest_planned_dose"}
    assert np.all(safe.rolling_six.selection_probability == [0, 0, 1])
    assert not safe.true_toxicity.flags.writeable
    report = tite_boin_rolling_six_report(safe)
    assert "highest_planned_dose" in report
    assert "not established MTD claims" in report
    assert "require_complete_before_escalation=True" in report


def test_comparison_replays_from_seed_and_preflights_before_rng_use() -> None:
    kwargs = dict(
        design=BOINDesign(target=0.25),
        true_toxicity=[0.1, 0.25],
        window=2.0,
        accrual_rate=1.5,
        cohorts=2,
        cohort_size=2,
        rolling_six_max_patients=6,
        trials=2,
    )
    first = compare_tite_boin_rolling_six(**kwargs, rng=17)
    second = compare_tite_boin_rolling_six(**kwargs, rng=17)
    np.testing.assert_array_equal(first.tite_boin.patients, second.tite_boin.patients)
    np.testing.assert_array_equal(first.rolling_six.toxicities, second.rolling_six.toxicities)

    generator = np.random.default_rng(9)
    state = generator.bit_generator.state
    with pytest.raises(ValueError, match="completion fraction"):
        compare_tite_boin_rolling_six(**kwargs, rng=generator, minimum_complete_fraction=0.1)
    assert generator.bit_generator.state == state
