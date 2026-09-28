import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats.bacis_dic import bacis_classification_dic


def test_classification_dic_matches_all_independent_quadrature_rows():
    fixture = Path(__file__).parent / "fixtures" / "bacis-dic-reference.csv"
    cases = defaultdict(list)
    with fixture.open(newline="") as stream:
        for row in csv.DictReader(stream):
            cases[row["case"]].append(row)

    for rows in cases.values():
        result = bacis_classification_dic(
            [int(row["y"]) for row in rows],
            [int(row["n"]) for row in rows],
            phi_low=float(rows[0]["phi_low"]),
            phi_high=float(rows[0]["phi_high"]),
            classification_precision=float(rows[0]["precision"]),
        )
        for field, key in (
            (result.posterior_mean, "posterior_mean"),
            (result.posterior_logit_mean, "posterior_logit_mean"),
            (result.mean_deviance, "mean_deviance"),
            (result.penalty, "penalty"),
            (result.dic, "dic"),
        ):
            np.testing.assert_allclose(
                field, [float(row[key]) for row in rows], rtol=2e-9, atol=2e-9
            )
        assert result.total_dic == pytest.approx(sum(result.dic), rel=2e-14)
        assert not result.dic.flags.writeable
        assert np.isfinite(result.quadrature_error).all()
        assert np.max(result.quadrature_error) < 2e-7


def test_separated_component_mixture_retains_between_component_penalty():
    result = bacis_classification_dic(
        [5], [10], phi_low=0.01, phi_high=0.99, classification_precision=100
    )
    assert result.posterior_mean[0] == pytest.approx(0.5, abs=2e-12)
    assert result.posterior_logit_mean[0] == pytest.approx(0.0, abs=2e-12)
    assert result.high_component_probability[0] == pytest.approx(0.5, abs=2e-12)
    # The large pD is primarily the between-component covariance contribution.
    assert result.penalty[0] > 20
    assert result.total_dic == pytest.approx(56.8673826585314, rel=2e-9)
