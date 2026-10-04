import numpy as np
import pytest
from scipy.special import betainc

from mdanderson_stats.uboin_conduct import UBOINDesign
from mdanderson_stats.uboin_imputation import uboin_stage2_multiple_imputation


def _design(**overrides: object) -> UBOINDesign:
    values: dict[str, object] = {
        "prior": [[0.25, 0.25], [0.25, 0.25]],
        "utilities": [[0, 30], [50, 100]],
        "candidate_scope": "tried",
        "s1": 3,
        "s2": 20,
        "max_patients": 20,
        "efficacy_cutoff": 0.999,
        "safety_cutoff": 0.999,
    }
    values.update(overrides)
    return UBOINDesign(**values)  # type: ignore[arg-type]


def test_zero_and_one_predictive_probabilities_are_exact_completions() -> None:
    design = _design()
    observed = np.zeros((2, 2, 2), dtype=int)
    observed[0, 0, 0] = 3
    result = uboin_stage2_multiple_imputation(
        design,
        observed,
        [1, 1],
        [1, 1],
        np.array([[0.0, 1.0]] * 5),
        current_dose=1,
        rng=np.random.default_rng(10),
    )
    assert np.array_equal(result.imputed_response, [[False, True]] * 5)
    assert np.array_equal(
        result.mean_utility_by_imputation, np.repeat(result.mean_utility[None], 5, axis=0)
    )
    assert np.array_equal(
        result.low_efficacy_probability_by_imputation,
        np.repeat(result.low_efficacy_probability[None], 5, axis=0),
    )
    assert not result.imputed_response.flags.writeable
    assert not result.mean_utility.flags.writeable


def test_mi_averages_complete_data_functionals_not_pooled_counts() -> None:
    design = _design(efficacy_limit=0.35)
    observed = np.zeros((1, 2, 2), dtype=int)
    observed[0, 0, 0] = 2
    probabilities = np.array([[0.0], [0.0], [1.0], [1.0], [0.0]])
    result = uboin_stage2_multiple_imputation(
        design,
        observed,
        [1],
        [1],
        probabilities,
        current_dose=1,
        rng=np.random.default_rng(11),
    )
    assert np.array_equal(result.imputed_response[:, 0], [False, False, True, True, False])
    expected = []
    for response in result.imputed_response[:, 0]:
        complete = observed.copy()
        complete[0, int(response), 1] += 1
        expected.append(design._posterior(complete).low_efficacy_probability[0])
    assert result.low_efficacy_probability[0] == pytest.approx(np.mean(expected), rel=0, abs=1e-15)

    prior = np.asarray(design.prior)
    response_mass = prior[1, :].sum() + 2 / 5
    nonresponse_mass = prior[0, :].sum() + 2 + 3 / 5
    pooled_tail = betainc(response_mass, nonresponse_mass, design.efficacy_limit)
    assert result.low_efficacy_probability[0] != pytest.approx(pooled_tail, rel=0, abs=1e-5)
    pending_toxicity_counts = observed + np.array([[[1, 0], [0, 0]]])
    assert np.array_equal(
        result.overdose_probability, design._posterior(pending_toxicity_counts).overdose_probability
    )


def test_zero_pending_matches_complete_stage_two_controller() -> None:
    design = _design()
    observed = np.zeros((2, 2, 2), dtype=int)
    observed[0, 0, 0] = 3
    direct = design.decision(observed, current_dose=1, stage=2)
    mi = design.decision_multiple_imputation(
        observed,
        [],
        [],
        np.empty((5, 0)),
        current_dose=1,
        rng=np.random.default_rng(12),
    )
    assert (mi.action, mi.next_dose, mi.selected_dose) == (
        direct.action,
        direct.next_dose,
        direct.selected_dose,
    )
    assert np.array_equal(mi.allocation_probabilities, direct.allocation_probabilities)
    assert np.array_equal(mi.mean_utility, direct.posterior.mean_utility)
    assert np.array_equal(mi.low_efficacy_probability, direct.posterior.low_efficacy_probability)


def test_pending_patients_count_for_cap_and_stage_two_b1_toxicity_rate() -> None:
    observed = np.zeros((2, 2, 2), dtype=int)
    observed[0, 0, 0] = 3
    capped = uboin_stage2_multiple_imputation(
        _design(max_patients=5),
        observed,
        [1, 1],
        [1, 1],
        np.zeros((5, 2)),
        current_dose=1,
        rng=np.random.default_rng(13),
    )
    assert capped.action == "stop_max_patients"

    changed_rate = uboin_stage2_multiple_imputation(
        _design(),
        observed,
        [1, 1],
        [2, 2],
        np.zeros((5, 2)),
        current_dose=1,
        rng=np.random.default_rng(14),
    )
    assert _design().decision(observed, current_dose=1, stage=2).action == "escalate"
    assert changed_rate.action != "escalate"
    expected_toxicity = observed.copy()
    expected_toxicity[0, 0, 1] += 2
    assert np.array_equal(
        changed_rate.overdose_probability,
        _design()._posterior(expected_toxicity).overdose_probability,
    )
    safety_stop = uboin_stage2_multiple_imputation(
        _design(max_patients=5),
        observed,
        [1, 1],
        [1, 1],
        np.zeros((5, 2)),
        current_dose=1,
        eliminated=[True, False],
        rng=np.random.default_rng(16),
    )
    assert safety_stop.action == "stop_safety"


def test_invalid_predictive_shape_fails_before_randomness() -> None:
    design = _design()
    observed = np.zeros((2, 2, 2), dtype=int)
    observed[0, 0, 0] = 1
    rng = np.random.default_rng(15)
    state = rng.bit_generator.state
    with pytest.raises(ValueError, match="rectangular"):
        uboin_stage2_multiple_imputation(
            design,
            observed,
            [1],
            [1],
            [[0.5], [0.5, 0.5], [0.5], [0.5], [0.5]],
            current_dose=1,
            rng=rng,
        )
    assert rng.bit_generator.state == state
