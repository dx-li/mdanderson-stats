import numpy as np
import pytest

from mdanderson_stats.mds_hope import (
    MDSHopeCovariates,
    mds_hope_risk_groups,
    mds_hope_score,
    mds_hope_standardized_risk_groups,
)


def _patient(**overrides):
    values = {
        "age_years": 60,
        "anc_10e9_l": 2.0,
        "hemoglobin_g_dl": 10.0,
        "platelets_10e9_l": 100.0,
        "marrow_blast_percent": 5.0,
        "cytogenetic_risk_score": 2.0,
        "sf3b1_mutation": 1,
        "ezh2_mutation": 1,
        "tp53_hit_count": 2,
        "kras_mutation": 1,
        "ptpn11_mutation": 1,
    }
    values.update(overrides)
    return MDSHopeCovariates(**values)


def test_eq_s1_contributions_use_raw_units_and_preserve_anc_sign():
    result = mds_hope_score(_patient())
    expected = np.asarray(
        [[1.5, 0.134, -1.6, -0.21, 0.255, 0.568, -0.294, 0.347, 0.69, 0.681, 1.094]]
    )
    np.testing.assert_allclose(result.contributions, expected, rtol=0, atol=1e-14)
    np.testing.assert_allclose(result.linear_predictor, expected.sum(axis=1), rtol=0, atol=1e-14)
    assert result.contribution_names[1] == "anc_10e9_l"
    assert result.contributions[0, 1] > 0
    assert not result.contributions.flags.writeable
    assert not result.linear_predictor.flags.writeable
    assert result.relative_hazard is None


def test_batches_broadcast_scalars_and_reference_profile_yields_log_hr():
    batch = _patient(age_years=[60, 65], sf3b1_mutation=[0, 1], tp53_hit_count=[0, 2])
    reference = _patient(age_years=60, sf3b1_mutation=0, tp53_hit_count=0)
    result = mds_hope_score(batch, reference=reference)
    baseline_eta = mds_hope_score(reference).linear_predictor[0]
    np.testing.assert_allclose(result.log_hazard_ratio, result.linear_predictor - baseline_eta)
    np.testing.assert_allclose(result.relative_hazard, np.exp(result.log_hazard_ratio))
    assert result.contributions.shape == (2, 11)
    assert not result.log_hazard_ratio.flags.writeable
    assert not result.relative_hazard.flags.writeable


def test_relative_hazard_underflow_keeps_log_hr_and_overflow_is_rejected():
    baseline = _patient(age_years=0, ptpn11_mutation=0)
    low = mds_hope_score(
        _patient(age_years=0, hemoglobin_g_dl=1e308, ptpn11_mutation=0), reference=baseline
    )
    assert np.isfinite(low.log_hazard_ratio[0])
    assert low.relative_hazard[0] == 0
    with pytest.raises(ArithmeticError, match="overflows"):
        mds_hope_score(
            _patient(age_years=40_000, ptpn11_mutation=0),
            reference=_patient(age_years=0, ptpn11_mutation=0),
        )


def test_log_hazard_ratio_uses_raw_differences_before_large_score_summation():
    quiet = {
        "anc_10e9_l": 0,
        "hemoglobin_g_dl": 0,
        "platelets_10e9_l": 0,
        "marrow_blast_percent": 0,
        "cytogenetic_risk_score": 0,
        "sf3b1_mutation": 0,
        "ezh2_mutation": 0,
        "tp53_hit_count": 0,
        "kras_mutation": 0,
        "ptpn11_mutation": 0,
    }
    reference = _patient(age_years=1e14, **quiet)
    patient = _patient(age_years=1e14 + 1, **quiet)
    result = mds_hope_score(patient, reference=reference)
    assert result.log_hazard_ratio[0] == pytest.approx(0.025, abs=1e-14)


def test_published_six_group_cutoffs_and_explicit_standardization_paths():
    standardized = np.asarray([-1.6, -1.5, -1.0, -0.5, -0.1, 0, 0.1, 0.5, 1, 1.5, 1.6])
    expected = np.asarray([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5], dtype=np.int8)
    direct = mds_hope_risk_groups(standardized)
    np.testing.assert_array_equal(direct, expected)
    assert not direct.flags.writeable

    calibrated = mds_hope_standardized_risk_groups(
        standardized * 3 + 10, reference_center=10, reference_sd=3
    )
    np.testing.assert_allclose(calibrated.standardized_score, standardized)
    np.testing.assert_array_equal(calibrated.group_code, expected)
    assert calibrated.group_labels[5] == "very high"
    with pytest.raises(ValueError, match="reference_sd"):
        mds_hope_standardized_risk_groups([1], reference_center=0, reference_sd=0)


def test_invalid_missing_or_misencoded_inputs_fail_without_imputation():
    with pytest.raises(ValueError, match="sf3b1_mutation"):
        mds_hope_score(_patient(sf3b1_mutation=None))
    with pytest.raises(ValueError, match="only 0 or 1"):
        mds_hope_score(_patient(kras_mutation=2))
    with pytest.raises(ValueError, match="0, 1, or 2"):
        mds_hope_score(_patient(tp53_hit_count=1.5))
    with pytest.raises(ValueError, match="same length"):
        mds_hope_score(_patient(age_years=[50, 60], anc_10e9_l=[1, 2, 3]))
    with pytest.raises(ValueError, match="cytogenetic_risk_score"):
        mds_hope_score(_patient(cytogenetic_risk_score="poor"))


def test_batch_limit_is_checked_before_building_the_covariate_matrix():
    too_many = np.zeros(250_001)
    with pytest.raises(ValueError, match="batch exceeds 250000"):
        mds_hope_score(_patient(age_years=too_many))
