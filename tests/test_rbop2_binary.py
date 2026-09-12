import pytest

from mdanderson_stats.rbop2_binary import rbop2_binary_design


def test_exact_inclusive_final_boundary():
    d = rbop2_binary_design(
        [[1, 1], [2, 2]],
        prior=[[1, 1], [1, 1]],
        endpoint="efficacy",
        margin=0,
        lower_cutoffs=[0.25, 0.8],
        upper_cutoffs=[0.8, 0.8],
    )
    assert d.monitor(1, 0, [1, 1]).probability == pytest.approx(5 / 6)
    assert d.monitor(1, 0, [1, 1]).superior
    assert d.monitor(0, 1, [1, 1]).futile


def test_toxicity_reverses_ordering():
    d = rbop2_binary_design(
        [[1, 1]],
        prior=[[1, 1], [1, 1]],
        endpoint="toxicity",
        margin=0,
        lower_cutoffs=[0.8],
        upper_cutoffs=[0.8],
    )
    assert d.monitor(0, 1, [1, 1]).superior
    assert d.monitor(1, 0, [1, 1]).futile


def test_boundary_table_and_exact_oc_absorb_early_stops():
    d = rbop2_binary_design(
        [[1, 1], [2, 2]],
        prior=[[1, 1], [1, 1]],
        endpoint="efficacy",
        margin=0,
        lower_cutoffs=[0.25, 0.8],
        upper_cutoffs=[0.8, 0.8],
    )
    table = d.boundary_table()
    assert (1, 0) in [tuple(x) for x in table.looks[0].superior_events]
    oc = d.operating_characteristics(0.3, 0.3)
    assert oc.stop_futility.shape == (2,)
    assert float(oc.final_positive) == pytest.approx(0.1218, abs=2e-4)
    assert float(oc.overall_positive) == pytest.approx(0.3318, abs=2e-4)
    assert float(oc.expected_total_sample_size) == pytest.approx(3.16, abs=2e-3)


def test_validation_rejects_nonpaired_or_final_gap():
    with pytest.raises(ValueError):
        rbop2_binary_design(
            [[1, 1], [1, 2]],
            prior=[[1, 1], [1, 1]],
            endpoint="efficacy",
            margin=0,
            lower_cutoffs=[0.2, 0.2],
            upper_cutoffs=[0.8, 0.9],
        )
    with pytest.raises(ValueError):
        rbop2_binary_design(
            [[1, 1]],
            prior=[[1, 1], [1, 1]],
            endpoint="efficacy",
            margin=0,
            lower_cutoffs=[0.2],
            upper_cutoffs=[0.8],
        )


def test_off_schedule_monitor_continues_and_unequal_arms_are_supported():
    design = rbop2_binary_design(
        [[2, 1], [4, 2]],
        prior=[[1, 1], [1, 1]],
        endpoint="efficacy",
        margin=0,
        lower_cutoffs=[0.2, 0.8],
        upper_cutoffs=[0.8, 0.8],
    )
    result = design.monitor(1, 0, [3, 1])
    assert result.decision == "continue"


def test_signed_margin_with_fractional_prior_is_evaluated():
    design = rbop2_binary_design(
        [[2, 2]],
        prior=[[0.7, 1.3], [1.2, 0.8]],
        endpoint="toxicity",
        margin=-0.2,
        lower_cutoffs=[0.8],
        upper_cutoffs=[0.8],
    )
    result = design.monitor(1, 0, [2, 2])
    assert 0 <= result.probability <= 1
