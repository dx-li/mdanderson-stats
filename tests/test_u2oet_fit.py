import numpy as np
from numpy.testing import assert_allclose

from mdanderson_stats import fit_u2oet, summarize_chains, u2oet_parameter_names


def test_fitted_posterior_matches_independent_r_integrals():
    names = u2oet_parameter_names(2, 2, model="cmi")
    mean = np.zeros(len(names) - 1)
    sd = np.full(mean.size, 1e-10)
    for i, name in enumerate(names[:-1]):
        if ".slope." in name:
            mean[i] = 1
    mean[names.index("toxicity.intercept.1")] = -1000
    sd[0] = 1
    counts = np.zeros((3, 3, 2, 2))
    counts[1, 1, 0, 0] = 7
    counts[1, 1, 1, 0] = 3
    fit = fit_u2oet(
        [1, 2, 3],
        [1, 2, 3],
        counts,
        prior_mean=mean,
        prior_sd=sd,
        model="cmi",
        draws=1500,
        warmup=300,
        chains=4,
        rng=np.random.default_rng(7701),
    )
    alpha = summarize_chains(fit.parameters[..., 0])
    efficacy = summarize_chains(fit.joint[:, :, 1, 1, 1].sum(axis=-1))
    # R stats::integrate: normal(alpha;0,1)*logistic(alpha)^3*logistic(-alpha)^7.
    reference_alpha = -0.61233147641579189
    reference_probability = 0.36123314764157921
    assert abs(alpha.mean - reference_alpha) < 5 * alpha.batch_mean_mcse
    assert abs(efficacy.mean - reference_probability) < 5 * efficacy.batch_mean_mcse
    assert alpha.split_rhat < 1.04
    assert_allclose(fit.joint.sum(axis=(-2, -1)), 1, atol=2e-14)
    assert np.all(fit.association_acceptance > 0.99)
    assert not fit.parameters.flags.writeable


def test_empty_data_recovers_truncated_normal_and_log_coordinate_priors():
    names = u2oet_parameter_names(2, 2)
    mean = np.zeros(len(names) - 1)
    sd = np.full(mean.size, 0.3)
    fit = fit_u2oet(
        [1, 2],
        [1, 2],
        np.zeros((2, 2, 2, 2)),
        prior_mean=mean,
        prior_sd=sd,
        draws=1500,
        warmup=300,
        chains=2,
        rng=np.random.default_rng(7702),
    )
    summary = summarize_chains(fit.parameters)
    expected = np.r_[mean, 0.0]
    for i, name in enumerate(names):
        if ".slope." in name:
            expected[i] = 0.3 * np.sqrt(2 / np.pi)
            assert np.all(fit.parameters[..., i] > 0)
    assert np.all(np.abs(summary.mean - expected) < 6 * summary.batch_mean_mcse)
    assert_allclose(fit.log_likelihood, 0)
    assert_allclose(fit.association_acceptance, 1)


def test_full_pds_posterior_matches_independent_r_importance_sampling():
    from pathlib import Path

    reference = np.loadtxt(
        Path(__file__).parent / "fixtures/u2oet-posterior-importance.csv",
        skiprows=1,
        delimiter=",",
    )
    mean, sd = np.zeros(12), np.full(12, 0.2)
    mean[[0, 6]], sd[[0, 6]] = [-0.3, -0.8], 0.3
    mean[[1, 2, 7, 8]] = 0.5
    counts = np.zeros((3, 3, 2, 2))
    for cell in ((0, 0, 0, 0), (1, 1, 1, 0), (2, 2, 1, 1), (0, 2, 1, 0)):
        counts[cell] = 1
    fit = fit_u2oet(
        [1, 2, 3],
        [1, 2, 3],
        counts,
        prior_mean=mean,
        prior_sd=sd,
        coordinate_updates=True,
        draws=1500,
        warmup=300,
        chains=4,
        rng=np.random.default_rng(7704),
    )
    summary = summarize_chains(fit.parameters)
    combined = np.sqrt(summary.batch_mean_mcse**2 + reference[:, 1] ** 2)
    assert np.all(np.abs(summary.mean - reference[:, 0]) < 6 * combined)
    assert np.all(summary.split_rhat < 1.05)


def test_joint_link_move_is_reversible_with_the_correct_density_jacobian():
    from mdanderson_stats.u2oet import _marginal
    from mdanderson_stats.u2oet_fit import _link_move, _parameters

    for model in ("pds", "cmi", "pds+cmi"):
        x = np.array(
            [-0.4, 0.8, 0.4, 0.5, 0.7, 0.9]
            + ([0.0, 0.0] if model != "cmi" else [])
            + [0.3]
            + ([0.2] if model != "pds" else [])
        )
        y, log_jac = _link_move(x, 3, model, 1.2)
        back, inverse_jac = _link_move(y, 3, model, -1.2)
        assert_allclose(back, x, atol=2e-14)
        assert_allclose(log_jac + inverse_jac, 0, atol=2e-14)
        h = 1e-5
        columns = []
        for j in range(x.size):
            direction = np.eye(x.size)[j] * h
            columns.append(
                (
                    _link_move(x + direction, 3, model, 1.2)[0]
                    - _link_move(x - direction, 3, model, 1.2)[0]
                )
                / (2 * h)
            )
        sign, numeric_jac = np.linalg.slogdet(np.column_stack(columns))
        assert sign == 1
        assert_allclose(log_jac, numeric_jac, atol=2e-9)
        original = _marginal([1, 2, 3], [1, 2, 3], _parameters(x, 3, model), "log")
        proposed = _marginal([1, 2, 3], [1, 2, 3], _parameters(y, 3, model), "log")
        assert_allclose(original[1, 1], proposed[1, 1], atol=2e-14)
