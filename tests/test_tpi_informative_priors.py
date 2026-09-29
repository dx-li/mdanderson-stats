import numpy as np
import pytest

from mdanderson_stats.tpi import TPIDesign
from mdanderson_stats.tpi_simulation import simulate_tpi


def test_dose_specific_beta_shapes_follow_final_axis_and_explicit_dose():
    priors = np.array([[1.0, 9.0], [2.0, 6.0], [3.0, 7.0]])
    design = TPIDesign(prior=priors)
    patients = np.array([4, 5, 6])
    toxicities = np.array([0, 2, 3])
    result = design.posterior(patients, toxicities)
    expected_mean = (toxicities + priors[:, 0]) / (patients + priors.sum(axis=1))
    np.testing.assert_allclose(result.mean, expected_mean)
    for dose in range(1, 4):
        selected = design.posterior(patients[dose - 1], toxicities[dose - 1], dose=dose)
        np.testing.assert_allclose(selected.mean, expected_mean[dose - 1])


def test_per_dose_prior_axis_and_decision_table_require_explicit_shape():
    design = TPIDesign(prior=[[1, 9], [2, 8], [3, 7]])
    batched = design.posterior([[0, 0, 0], [2, 2, 2]], [[0, 0, 0], [0, 1, 2]])
    assert batched.mean.shape == (2, 3)
    with pytest.raises(ValueError, match="last axis"):
        design.posterior([1, 1], [0, 0])
    with pytest.raises(ValueError, match="dose="):
        design.posterior(1, 0)
    with pytest.raises(ValueError, match="dose="):
        design.decision_table(6)
    assert design.decision_table(6, dose=2).action.shape == (7, 6)


def test_common_prior_path_remains_common_and_rejects_complex_values():
    common = TPIDesign(prior=(0.005, 0.005))
    posterior = common.posterior([3, 6], [0, 1])
    np.testing.assert_allclose(posterior.mean, [(0.005) / 3.01, 1.005 / 6.01])
    with pytest.raises(ValueError, match="prior shapes must be real"):
        TPIDesign(prior=np.array([1 + 0j, 2]))
    with pytest.raises(ValueError, match="counts must be real"):
        common.posterior(np.array([3 + 0j]), [0])


def test_simulation_uses_per_dose_priors_and_requires_matching_dose_count():
    design = TPIDesign(prior=[[1, 9], [2, 8], [3, 7]])
    result = simulate_tpi(
        design,
        [0.1, 0.25, 0.4],
        cohorts=5,
        cohort_size=1,
        trials=20,
        rng=4,
    )
    assert result.patients.shape == (20, 3)
    assert np.all(result.toxicities <= result.patients)
    with pytest.raises(ValueError, match="match the dose-specific"):
        simulate_tpi(design, [0.1, 0.25], cohorts=5, cohort_size=1, trials=2, rng=4)
    with pytest.raises(ValueError, match="must be real"):
        simulate_tpi(design, np.array([0.1 + 0j, 0.25, 0.4]), trials=2)
