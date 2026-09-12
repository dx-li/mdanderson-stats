import numpy as np
import pytest

from mdanderson_stats.bacis import bacis_classify, bacis_fit


def test_classification_symmetry_and_ordering() -> None:
    result = bacis_classify([5, 9], [10, 10], phi_low=0.2, phi_high=0.8)
    assert result.high_probability[0] == pytest.approx(1 - result.low_probability[0])
    assert result.high_probability[1] > result.high_probability[0]
    assert result.cluster[1] == 2
    assert np.all(result.quadrature_error < 1e-7)


def test_adaptive_cutoff_exposes_native_weighting_variants() -> None:
    subgroup = bacis_classify(
        [0, 10], [1, 100], classification_cutoff=None, adaptive_weighting="subgroup"
    )
    patient = bacis_classify(
        [0, 10], [1, 100], classification_cutoff=None, adaptive_weighting="patient"
    )
    assert subgroup.cutoff != patient.cutoff
    with pytest.raises(ValueError):
        bacis_classify([1], [2], adaptive_weighting="invalid")
    with pytest.raises(ValueError):
        bacis_classify([1], [2], classification_cutoff=0.5, adaptive_weighting="invalid")


def test_singleton_cluster_uses_exact_beta_summary() -> None:
    result = bacis_fit(
        [0, 8], [10, 10], phi_low=0.1, phi_high=0.3, draws=16, warmup=4, chains=2, seed=3
    )
    # The first subgroup is low and isolated; its analytical Beta(1,11) moments
    # are reported even though probability_samples are generated for diagnostics.
    assert result.posterior_mean[0] == pytest.approx(1 / 12)
    assert result.posterior_sd[0] == pytest.approx(np.sqrt(11 / (12**2 * 13)))
    assert result.cluster_fits[0] is None or result.cluster_fits[1] is None
    assert result.probability_samples.shape == (2, 16, 2)


def test_two_cluster_fit_reproducible_and_diagnostics_finite() -> None:
    kwargs = dict(draws=16, warmup=8, chains=2, seed=11, classification_cutoff=0.5)
    first = bacis_fit([0, 1, 9, 10], [10, 10, 10, 10], **kwargs)
    second = bacis_fit([0, 1, 9, 10], [10, 10, 10, 10], **kwargs)
    assert np.array_equal(first.probability_samples, second.probability_samples)
    assert first.probability_samples.shape == (2, 16, 4)
    assert np.all(np.isfinite(first.summary.mean))
    assert np.all(np.isfinite(first.posterior_sd))


def test_guards_and_readonly_outputs() -> None:
    with pytest.raises(ValueError):
        bacis_classify(np.zeros(101), np.ones(101))
    with pytest.raises(ValueError):
        bacis_classify([2], [1])
    with pytest.raises(ValueError):
        bacis_fit([1, 2], [2, 2], draws=7)
    result = bacis_classify([1], [2])
    assert not result.high_probability.flags.writeable
    assert not result.cluster.flags.writeable
