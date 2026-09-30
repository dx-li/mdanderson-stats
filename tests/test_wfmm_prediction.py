import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats.hierarchical_binomial import summarize_chains
from mdanderson_stats.wfmm_basis import wfmm_basis, wfmm_inverse
from mdanderson_stats.wfmm_model import WFMMCoefficientFit
from mdanderson_stats.wfmm_posterior import wfmm_summarize
from mdanderson_stats.wfmm_prediction import wfmm_predict_coefficients


def _fit(*, random_effects=True, random_variance=4.0, residual_variance=0.25, draws=8):
    chains, fixed, levels, coefficients = 2, 2, 2, 2
    beta = np.empty((chains, draws, fixed, coefficients))
    beta[:, :, 0, :] = np.arange(chains * draws).reshape(chains, draws, 1) / 10
    beta[:, :, 1, :] = 1.0
    random = np.zeros((chains, draws, levels, coefficients))
    random[:, :, 0, :] = 2.0
    random[:, :, 1, :] = -1.0
    return WFMMCoefficientFit(
        coefficients=beta,
        inclusion_indicators=np.ones_like(beta, dtype=bool),
        random_variances=np.full((chains, draws, 1, coefficients), random_variance),
        residual_variances=np.full((chains, draws, 2, coefficients), residual_variance),
        random_effects=random if random_effects else None,
        log_likelihood=np.zeros((chains, draws)),
        summary=summarize_chains(beta),
        random_strata=np.zeros(levels, dtype=np.int64),
        residual_strata=np.array([0, 1], dtype=np.int64),
        coefficient_partition=np.zeros(coefficients, dtype=np.int64),
        coefficient_scale=np.zeros(coefficients, dtype=np.int64),
        warmup=0,
        estimate_variances=False,
        variance_acceptance=np.zeros((3, coefficients)),
        likelihood_evaluations=0,
    )


def test_fixed_and_existing_random_latent_coefficient_draws_are_exact():
    fit = _fit()
    x = np.array([[1.0, 2.0], [-1.0, 0.5]])
    z = np.array([[1.0, 0.0], [0.25, 1.0]])
    population = wfmm_predict_coefficients(fit, x)
    assert_allclose(population, np.einsum("rp,cdpk->cdrk", x, fit.coefficients))
    result = wfmm_predict_coefficients(fit, x, existing_random_design=z)
    expected = np.einsum("rp,cdpk->cdrk", x, fit.coefficients)
    expected += np.einsum("rm,cdmk->cdrk", z, fit.random_effects)
    assert_allclose(result, expected)
    assert result.shape == (2, 8, 2, 2)
    assert not result.flags.writeable

    # The coefficient output is directly consumable by the existing summary path.
    basis = wfmm_basis(
        2,
        transform="custom",
        analysis_matrix=[[1.0, -0.5], [0.0, 0.5]],
        synthesis_matrix=[[1.0, 1.0], [0.0, 2.0]],
    )
    summary = wfmm_summarize(result, basis)
    assert summary.mean.shape == (2, 2)
    reconstructed = wfmm_inverse(result.reshape(-1, 2), basis).reshape(result.shape)
    assert_allclose(summary.mean, np.mean(reconstructed, axis=(0, 1)))


def test_new_level_is_shared_across_rows_and_generator_is_required():
    fit = _fit(random_effects=False)
    fixed = np.zeros((2, 2))
    new_design = np.array([[1.0], [2.0]])
    with pytest.raises(ValueError, match="explicit Generator"):
        wfmm_predict_coefficients(
            fit,
            fixed,
            new_random_design=new_design,
            new_random_strata=[0],
        )
    result = wfmm_predict_coefficients(
        fit,
        fixed,
        new_random_design=new_design,
        new_random_strata=[0],
        rng=np.random.default_rng(21),
    )
    assert_allclose(result[:, :, 1, :], 2.0 * result[:, :, 0, :])
    replay = wfmm_predict_coefficients(
        fit,
        fixed,
        new_random_design=new_design,
        new_random_strata=[0],
        rng=np.random.default_rng(21),
    )
    assert_allclose(result, replay)
    # Each coefficient has known prior variance, estimated here over chain/draw.
    assert_allclose(np.var(result[:, :, 0, :], axis=(0, 1), ddof=1), 4.0, rtol=0.6)


def test_row_residuals_use_explicit_variance_groups_and_are_independent():
    fit = _fit(random_effects=False, residual_variance=1.0, draws=64)
    result = wfmm_predict_coefficients(
        fit,
        np.zeros((2, 2)),
        include_residual=True,
        residual_strata=[0, 1],
        rng=np.random.default_rng(22),
    )
    assert result.shape == (2, 64, 2, 2)
    for coefficient in range(2):
        row0 = result[:, :, 0, coefficient].ravel()
        row1 = result[:, :, 1, coefficient].ravel()
        assert abs(np.var(row0, ddof=1) - 1.0) < 1.0
        assert abs(np.var(row1, ddof=1) - 1.0) < 1.0
        assert abs(np.cov(row0, row1, ddof=1)[0, 1]) < 0.3


def test_bad_mappings_and_work_caps_fail_before_rng_use():
    fit = _fit(random_effects=False)
    rng = np.random.default_rng(23)
    state = rng.bit_generator.state
    with pytest.raises(ValueError, match="new_random_strata"):
        wfmm_predict_coefficients(
            fit,
            np.zeros((2, 2)),
            new_random_design=np.ones((2, 1)),
            new_random_strata=[1],
            rng=rng,
        )
    assert rng.bit_generator.state == state
    with pytest.raises(ValueError, match="max_work"):
        wfmm_predict_coefficients(
            fit,
            np.zeros((2, 2)),
            include_residual=True,
            residual_strata=[0, 1],
            max_work=1,
            rng=rng,
        )
    assert rng.bit_generator.state == state


def test_residual_strata_are_rejected_when_residual_is_not_requested():
    fit = _fit(random_effects=False)
    with pytest.raises(ValueError, match="only when include_residual"):
        wfmm_predict_coefficients(fit, np.zeros((2, 2)), residual_strata=[0, 1])
