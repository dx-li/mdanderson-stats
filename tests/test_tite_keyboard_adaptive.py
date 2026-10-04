import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad
from scipy.special import betainc, betaln, gammaln, logsumexp

from mdanderson_stats import KeyboardDesign, tite_keyboard_decision
from mdanderson_stats.tite_keyboard_adaptive import (
    _dose_beta_integrals,
    _log_marginal_dose,
    tite_keyboard_adaptive_decision,
    tite_keyboard_adaptive_weights,
)


def test_positive_beta_mixture_matches_direct_nuisance_probability_integral() -> None:
    lam, gam = 2.2, 1.3
    followup = np.array([0.2, 0.7])
    survival = 1 - betainc(lam, gam, followup)
    expected = quad(
        lambda p: p * (1 - p) * np.prod((1 - p) + p * survival),
        0,
        1,
        epsabs=1e-13,
    )[0]
    observed = np.exp(_log_marginal_dose(lam, gam, followup, _dose_beta_integrals(1, 1, 2)))
    with_zero_time = np.exp(
        _log_marginal_dose(lam, gam, np.array([0.2, 0.7, 0.0]), _dose_beta_integrals(1, 1, 3))
    )
    assert abs(observed - expected) < 2e-13
    assert abs(with_zero_time - expected) < 2e-13


def _joint_grid_reference(node_count: int) -> np.ndarray:
    """Directly integrate p, lambda and gamma on bounded Gauss-Legendre grids."""
    nodes, weights = leggauss(node_count)
    z = (nodes + 1) * 15 - 24
    z_weights = weights * 15
    lambdas = np.exp(z)
    gammas = lambdas.copy()
    p_nodes, p_weights = leggauss(64)
    p = (p_nodes + 1) / 2
    log_p_weights = np.log(p_weights / 2)
    pending = np.array([0.2, 0.7])
    event_time = 0.35
    log_mass = np.empty((z.size, z.size))
    weight_values = np.empty((z.size, z.size, 2))
    for i, lam in enumerate(lambdas):
        for j, gam in enumerate(gammas):
            cdf = betainc(lam, gam, pending)
            log_p_integrand = np.log(p) + np.log1p(-p) + np.log1p(-p[:, None] * cdf).sum(axis=1)
            log_integral = logsumexp(log_p_weights + log_p_integrand)
            # Independent Gamma(shape=.5, rate=.5) priors in log coordinates.
            log_prior = (
                0.5 * np.log(lam)
                - 0.5 * lam
                + 0.5 * np.log(0.5)
                - gammaln(0.5)
                + 0.5 * np.log(gam)
                - 0.5 * gam
                + 0.5 * np.log(0.5)
                - gammaln(0.5)
            )
            log_event = (
                (lam - 1) * np.log(event_time)
                + (gam - 1) * np.log1p(-event_time)
                - betaln(lam, gam)
            )
            log_mass[i, j] = log_prior + log_event + log_integral
            weight_values[i, j] = cdf
    log_mass += np.log(z_weights)[:, None] + np.log(z_weights)[None, :]
    normalized = np.exp(log_mass - logsumexp(log_mass))
    return np.sum(normalized[..., None] * weight_values, axis=(0, 1))


def test_posterior_mean_weights_match_joint_grid_and_rescale_with_time_units() -> None:
    kwargs = dict(
        observed_dlt_times=[[], [3.5]],
        completed_nondlt=[2, 1],
        pending_followup=[[], [2.0, 7.0]],
        window=10.0,
        lambda_prior=(0.5, 0.5),
        gamma_prior=(0.5, 0.5),
        chains=4,
        draws=6_000,
        warmup=1_500,
        max_weight_mcse=0.05,
        max_split_rhat=1.2,
    )
    fitted = tite_keyboard_adaptive_weights(**kwargs, rng=np.random.default_rng(731))
    scaled = tite_keyboard_adaptive_weights(
        **{
            **kwargs,
            "observed_dlt_times": [[], [10.5]],
            "pending_followup": [[], [6.0, 21.0]],
            "window": 30.0,
        },
        rng=np.random.default_rng(731),
    )
    coarse = _joint_grid_reference(71)
    expected = _joint_grid_reference(91)
    quadrature_refinement = np.abs(expected - coarse)
    discrepancy = np.abs(fitted.pending_weights[1] - expected)
    allowance = 4 * fitted.weight_mcse + 2 * quadrature_refinement + 0.002
    assert np.all(discrepancy <= allowance)
    np.testing.assert_allclose(fitted.pending_weights[1], scaled.pending_weights[1], atol=1e-14)
    assert fitted.diagnostics_passed
    assert fitted.evaluations == 4 * (1 + 2 * (6_000 + 1_500))
    assert np.all(np.isfinite(fitted.weight_mcse))


def test_existing_controller_accepts_adaptive_weights_and_wrapper_preflights() -> None:
    design = KeyboardDesign(target=0.3)
    ordinary = tite_keyboard_decision(
        design,
        [3, 3],
        [0, 1],
        [[], [2.0, 3.0]],
        current_dose=2,
        window=10.0,
        pending_fraction_limit=None,
        pending_weights=[[], [0.4, 0.6]],
    )
    expected = design._posterior_effective(np.asarray(2.0), np.asarray(1.0))
    np.testing.assert_allclose(ordinary.effective_sample_size, [3.0, 2.0])
    np.testing.assert_allclose(ordinary.posterior.probability, expected.probability)

    rng = np.random.default_rng(19)
    state = rng.bit_generator.state
    try:
        tite_keyboard_adaptive_decision(
            design,
            [3, 3],
            [0, 1],
            [[], []],
            [[], [2.0]],
            current_dose=2,
            window=10.0,
            lambda_prior=(0.5, 0.5),
            gamma_prior=(0.5, 0.5),
            rng=rng,
        )
    except ValueError as error:
        assert "count must equal" in str(error)
    else:
        raise AssertionError("mismatched event times should be rejected")
    assert rng.bit_generator.state == state

    try:
        tite_keyboard_adaptive_weights(
            [[], [3.5]],
            [2, 1],
            [[], [2.0, 7.0]],
            10.0,
            lambda_prior=(0.5, 0.5),
            gamma_prior=(0.5, 0.5),
            rng=rng,
            max_work=1,
        )
    except ValueError as error:
        assert "max_work" in str(error)
    else:
        raise AssertionError("insufficient work must be rejected")
    assert rng.bit_generator.state == state
