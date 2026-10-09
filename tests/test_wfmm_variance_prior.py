import json
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import fit_wfmm_coefficients, wfmm_variance_prior

REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/wfmm-native-variance-prior.json").read_text()
)


@pytest.mark.parametrize("reference", REFERENCE["cases"], ids=lambda row: row["case"])
def test_mapping_matches_original_linux_initializer(reference):
    omega = np.asarray(reference["omega_MLE"])
    levels = len(reference["random_effect_counts"])
    prior = wfmm_variance_prior(
        inclusion_probability=0.5,
        slab_variance=1.0,
        random_variance=omega[:levels] if levels else None,
        random_effect_counts=reference["random_effect_counts"] if levels else None,
        residual_variance=omega[levels:],
        residual_stratum_sizes=reference["residual_stratum_sizes"],
        delta_omega=reference["delta_omega"],
    )
    expected_a = np.asarray(reference["prior_omega_a"])
    expected_b = np.asarray(reference["prior_omega_b"])
    np.testing.assert_allclose(prior.residual_shape, expected_a[levels:], rtol=2e-15)
    np.testing.assert_allclose(prior.residual_scale, expected_b[levels:], rtol=2e-15)
    if levels:
        np.testing.assert_allclose(prior.random_shape, expected_a[:levels], rtol=2e-15)
        np.testing.assert_allclose(prior.random_scale, expected_b[:levels], rtol=2e-15)
    else:
        assert prior.random_shape is None and prior.random_scale is None


def test_default_prior_is_centered_in_precision_and_owns_input_arrays():
    omega = np.array([[2.0, 3.0]])
    prior = wfmm_variance_prior(
        inclusion_probability=0.5,
        slab_variance=1.0,
        residual_variance=omega,
        residual_stratum_sizes=[8],
    )
    np.testing.assert_allclose(prior.residual_shape, [[0.0008, 0.0008]])
    np.testing.assert_allclose(prior.residual_shape / prior.residual_scale, 1 / omega)
    assert np.all(prior.residual_shape < 1)  # Variance expectation does not exist.
    omega[:] = 99
    np.testing.assert_allclose(prior.residual_scale, [[0.0016, 0.0024]])
    assert not prior.residual_scale.flags.writeable


@pytest.mark.parametrize(
    "changes",
    [
        {"delta_omega": 0},
        {"delta_omega": -1},
        {"delta_omega": True},
        {"delta_omega": float("nan")},
        {"delta_omega": 1j},
        {"delta_omega": [1.0]},
        {"residual_stratum_sizes": [1.5]},
        {"residual_stratum_sizes": [True]},
        {"residual_stratum_sizes": [0]},
        {"residual_stratum_sizes": [501]},
        {"residual_stratum_sizes": [8, 1]},
        {"residual_variance": [1.0, 2.0]},
        {"residual_variance": [[0.0, 1.0]]},
        {"residual_variance": [[float("inf"), 1.0]]},
        {"residual_variance": [[1j, 1.0]]},
        {"residual_variance": np.ones((1, 513))},
        {"random_effect_counts": [4]},
        {"random_variance": [[1.0, 2.0]]},
        {"random_variance": [[1.0]], "random_effect_counts": [4]},
        {"random_variance": [[1.0, 2.0]], "random_effect_counts": [0]},
        {"random_variance": np.ones((2, 2)), "random_effect_counts": [300, 300]},
        {"residual_variance": np.ones((2, 2)), "residual_stratum_sizes": [300, 300]},
        {"delta_omega": 1e308, "residual_stratum_sizes": [8]},
        {"delta_omega": 1e-320, "residual_variance": [[1e-300, 1e-300]]},
    ],
)
def test_rejects_invalid_or_unrepresentable_parameters(changes):
    arguments = dict(
        inclusion_probability=0.5,
        slab_variance=1.0,
        residual_variance=[[1.0, 2.0]],
        residual_stratum_sizes=[8],
    )
    arguments.update(changes)
    with pytest.raises(ValueError):
        wfmm_variance_prior(**arguments)


def test_native_mapping_composes_with_estimated_variance_sampler():
    y = np.array([1.0, 2.0, 4.0, 5.0, 6.0, 3.0, 2.0, 4.0])[:, None]
    prior = wfmm_variance_prior(
        inclusion_probability=1.0,
        slab_variance=10.0,
        residual_variance=[[2.5]],
        residual_stratum_sizes=[8],
        delta_omega=0.2,
    )
    fit = fit_wfmm_coefficients(
        y,
        np.ones((8, 1)),
        prior=prior,
        residual_variance=2.5,
        proposal_sd=(None, 0.5),
        estimate_variances=True,
        draws=12,
        warmup=4,
        chains=2,
        rng=np.random.default_rng(807),
    )
    assert np.isfinite(fit.coefficients).all()
    assert np.all(fit.residual_variances > 0)
