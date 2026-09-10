import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import (
    U2OETCriteria,
    U2OETMarginal,
    fit_u2oet,
    read_u2oet_patients,
    summarize_chains,
    u2oet_next_patient,
    u2oet_patients,
    u2oet_posterior,
    u2oet_probabilities,
)


def test_pending_outcomes_keep_assignments_and_open_cohort_precedence(tmp_path):
    path = tmp_path / "patients.txt"
    path.write_text("1 1 1 1 0\n2 1 1 -1 1\n3 1 1 1 -1\n4 1 2 -1 -1\n")
    data = read_u2oet_patients(path, dose_counts=(2, 2), efficacy_levels=2, toxicity_levels=2)
    assert_allclose(data.treated, [[3, 1], [0, 0]])
    assert data.complete.sum() == 1 and data.toxicity_only.sum() == 1 and data.ignored_outcomes == 2
    assert data.last_dose == (0, 1) and data.trailing_treatments == 1
    marginal = U2OETMarginal([0], [[1, 1]])
    model = u2oet_probabilities(
        [1, 2], [1, 2], efficacy=marginal, toxicity=marginal, association=0.8
    )
    assert_allclose(
        model.loglikelihood(data.complete, toxicity_only=data.toxicity_only),
        model.log_joint[0, 0, 1, 0] + model.log_toxicity[0, 0, 1],
    )
    # Highest utility is at (1,1), but an acceptable open cohort stays at (0,1).
    p = np.zeros((1, 2, 2, 2, 2))
    p[0, :, :, 1, 0] = [[0.2, 0.1], [0.8, 0.9]]
    p[0, :, :, 0, 0] = 1 - p[0, :, :, 1, 0]
    criteria = U2OETCriteria(1, 1, min_efficacy=0, max_toxicity=1)
    summary = u2oet_posterior(p, [[0, 0], [100, 100]], criteria=criteria)
    decision = u2oet_next_patient(summary, data)
    assert decision.continuing_cohort and decision.cohort_position == 2
    assert decision.probabilities[0, 1] == 1
    stopped = u2oet_next_patient(summary, data, max_patients=4)
    assert stopped.probabilities.sum() == 0 and stopped.cohort_position == 0
    # Remove the open pair from acceptability; the next patient starts elsewhere.
    summary = u2oet_posterior(
        p, [[0, 0], [100, 100]], criteria=U2OETCriteria(1, 1, min_efficacy=0.15, max_toxicity=1)
    )
    decision = u2oet_next_patient(summary, data)
    assert decision.closed_unacceptable_cohort and not decision.continuing_cohort
    assert decision.cohort_position == 1 and decision.probabilities[1, 1] == 1
    # Six identical treatments close two cohorts, even when all outcomes are pending.
    records = [[i, 1, 1, -1, -1] for i in range(1, 7)]
    full = u2oet_patients(records, dose_counts=(2, 2), efficacy_levels=2, toxicity_levels=2)
    assert not u2oet_next_patient(summary, full).continuing_cohort
    with pytest.raises(ValueError, match="strictly increase"):
        u2oet_patients(
            [[1, 1, 1, 0, 0], [1, 1, 1, -1, -1]],
            dose_counts=(2, 2),
            efficacy_levels=2,
            toxicity_levels=2,
        )


def test_toxicity_only_posterior_matches_independent_logistic_normal_integral():
    mean, sd = np.zeros(12), np.full(12, 1e-10)
    mean[[1, 2, 7, 8]] = 1
    sd[6] = 1
    complete = np.zeros((3, 3, 2, 2))
    tox = np.zeros((3, 3, 2))
    tox[1, 1] = [7, 3]
    fit = fit_u2oet(
        [1, 2, 3],
        [1, 2, 3],
        complete,
        toxicity_only=tox,
        prior_mean=mean,
        prior_sd=sd,
        draws=1500,
        warmup=300,
        chains=4,
        rng=np.random.default_rng(7715),
    )
    summary = summarize_chains(fit.parameters[..., 6])
    # R integration already recorded in the U2OET fitting documentation.
    assert abs(summary.mean - (-0.61233147641579189)) < 5 * summary.batch_mean_mcse
    assert summary.split_rhat < 1.04
    assert_allclose(fit.association_acceptance, 1)
    observed = fit.joint[:, :, 1, 1].sum(axis=-2)
    assert_allclose(
        fit.log_likelihood, 7 * np.log(observed[..., 0]) + 3 * np.log(observed[..., 1]), atol=2e-13
    )
