import csv
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.special import expit

from mdanderson_stats.hierarchical_binomial import summarize_chains
from mdanderson_stats.mtadf_author_local import (
    _author_local_gate,
    _mtadf_author_local_decision_with_count,
    mtadf_author_local_posterior,
)

_FIXTURES = Path(__file__).parent / "fixtures"


def _rows(name: str) -> list[dict[str, str]]:
    with (_FIXTURES / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_adjacent_posterior_matches_independent_quadrature_references():
    for case_index, row in enumerate(_rows("mtadf-author-local-reference.csv")):
        dose_count = int(row["dose_count"])
        pair = json.loads(row["window_dose_indices"])
        pair_subjects = json.loads(row["window_subjects"])
        pair_responses = json.loads(row["window_responses"])
        subjects = np.zeros(dose_count, dtype=np.int64)
        responses = np.zeros(dose_count, dtype=np.int64)
        for source_index, n, y in zip(pair, pair_subjects, pair_responses, strict=True):
            subjects[source_index - 1] = n
            responses[source_index - 1] = y

        fit = mtadf_author_local_posterior(
            subjects,
            responses,
            lower_dose=pair[0] - 1,
            draws=8_000,
            warmup=2_000,
            chains=4,
            rng=np.random.default_rng(2718 + case_index),
        )
        reference_slope_probability = float(row["probability_positive_slope"])
        quadrature_error = float(
            row["quadrature_error_estimate"] or row["refinement_max_abs_error"]
        )
        slope_mcse = fit.probability_positive_slope_mcse
        slope_rhat = float(fit.positive_slope_summary.split_rhat)
        assert slope_mcse <= 0.03
        assert slope_rhat <= 1.1
        assert abs(fit.probability_positive_slope - reference_slope_probability) <= (
            6.0 * slope_mcse + quadrature_error
        )
        xs = (np.arange(1, dose_count + 1) - (dose_count + 1) / 2.0) / (
            2.0 * np.std(np.arange(1, dose_count + 1), ddof=1)
        )
        predicted = expit(
            fit.parameter_draws[:, :, 0, None]
            + fit.parameter_draws[:, :, 1, None] * xs[None, None, :]
        )
        prediction_summary = summarize_chains(predicted)
        reference_predictions = np.asarray(
            json.loads(row["posterior_mean_probabilities"]), dtype=float
        )
        prediction_mcse = np.asarray(prediction_summary.batch_mean_mcse, dtype=float)
        prediction_rhat = np.asarray(prediction_summary.split_rhat, dtype=float)
        assert np.max(prediction_mcse) <= 0.03
        assert np.max(prediction_rhat) <= 1.1
        assert np.all(
            np.abs(prediction_summary.mean - reference_predictions)
            <= 6.0 * prediction_mcse + quadrature_error
        )
        assert fit.positive_slope_draws.dtype == np.bool_
        assert fit.positive_slope_draws.shape == (4, 8_000)


def test_author_probability_gates_match_independent_source_ledger():
    for row in _rows("mtadf-author-local-gates.csv"):
        position = row["position"]
        current = int(row["dose"]) - 1
        backward = (
            None if row["p_backward_nonpositive"] == "" else float(row["p_backward_nonpositive"])
        )
        forward = None if row["p_forward_positive"] == "" else float(row["p_forward_positive"])
        if position == "safety_cap":
            continue
        desired, _ = _author_local_gate(
            current,
            5,
            backward_nonpositive=backward,
            forward_positive=forward,
            next_dose_observed=position == "interior_tried",
            ce1=float(row["ce1"]),
            ce2=float(row["ce2"]),
        )
        capped = min(desired, int(row["admissible_count"]) - 1)
        assert desired == int(row["desired_dose"]) - 1
        assert capped == int(row["capped_dose"]) - 1


def test_safety_shortcut_and_two_fit_preflight_preserve_rng():
    rng = np.random.default_rng(444)
    shortcut = _mtadf_author_local_decision_with_count(
        [3, 0, 0],
        [0, 0, 0],
        [1, 0, 0],
        current_dose=0,
        rng=rng,
        admissibility_count=1,
    )
    assert shortcut.dose == 0
    assert shortcut.fit_count == 0
    assert shortcut.admissible.dtype == np.bool_
    assert not shortcut.admissible.flags.writeable

    rng = np.random.default_rng(445)
    before = json.dumps(rng.bit_generator.state, sort_keys=True)
    with pytest.raises(ValueError, match="combined current and bounce posterior storage"):
        _mtadf_author_local_decision_with_count(
            [2, 2, 2],
            [0, 0, 0],
            [1, 1, 1],
            current_dose=1,
            draws=10_000,
            warmup=0,
            chains=16,
            rng=rng,
            admissibility_count=3,
        )
    assert json.dumps(rng.bit_generator.state, sort_keys=True) == before
