"""Streaming completeness and strict, provenance-bound portable reports."""

import csv
import io
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import read_uaroet_desktop_study
from mdanderson_stats.uaroet_desktop_conduct import recommend_uaroet_desktop
from mdanderson_stats.uaroet_precision import fit_uaroet_desktop
from mdanderson_stats.uaroet_reports import (
    UAROETDistributionWriter,
    uaroet_assignment_csv,
    uaroet_assignment_report,
)


def study(chains=2):
    s = read_uaroet_desktop_study(
        (Path(__file__).parent / "fixtures/uaroet-python-study.json").read_text()
    )
    return replace(s, simulation_parameters={"NumChains": chains, "BurnInSamples": 0})


def test_streaming_retains_every_marginal_and_utility_without_changing_chain():
    s = study()
    stream = io.StringIO()
    writer = UAROETDistributionWriter(stream, s)
    chunks = []

    def callback(c, n, fit):
        writer(c, n, fit)
        chunks.append(fit)

    options = dict(target=1e-6, chunk_draws=32, max_draws=64)
    fit = fit_uaroet_desktop(s, rng=np.random.default_rng(713), on_chunk=callback, **options)
    control = fit_uaroet_desktop(s, rng=np.random.default_rng(713), **options)
    np.testing.assert_array_equal(fit.utility_samples, control.utility_samples)
    writer.validate_complete(chains=2, draws_per_chain=64)
    rows = list(csv.DictReader(io.StringIO(stream.getvalue())))
    assert len(rows) == writer.rows_written == 2 * 64 * 2
    for row in rows:
        c, n, d = (int(row[k]) - 1 for k in ("chain_1based", "draw_1based", "dose_1based"))
        assert float(row["efficacy_p0"]) + float(row["efficacy_p1"]) == pytest.approx(1)
        assert float(row["toxicity_p0"]) + float(row["toxicity_p1"]) == pytest.approx(1)
        assert float(row["utility"]) == pytest.approx(fit.utility_samples[c, n, d], abs=1e-12)
    before = stream.getvalue()
    with pytest.raises(ValueError, match="contiguous"):
        writer(0, 0, chunks[0])
    assert stream.getvalue() == before
    with pytest.raises(ValueError, match="complete fit"):
        writer.validate_complete(chains=2, draws_per_chain=32)


def test_initial_report_has_input_provenance_and_no_invented_posterior():
    s = replace(study(), patients=())
    decision = recommend_uaroet_desktop(s, rng=np.random.default_rng(1))
    report = json.loads(uaroet_assignment_report(decision))
    assert report["input_study"] == json.loads(s.to_json())
    assert report["posterior"] is None and report["selected_dose_1based"] == 1
    assert report["clinical_validation"] is False
    rows = list(csv.DictReader(io.StringIO(uaroet_assignment_csv(decision))))
    assert [r["status"] for r in rows] == ["not_evaluated"] * 2
    assert all(r["mean_utility"] == "" for r in rows)


def test_one_chain_report_serializes_unavailable_rhat_as_null():
    s = study(1)
    decision = recommend_uaroet_desktop(
        s, rng=np.random.default_rng(54), target=0.99, chunk_draws=32, max_draws=32
    )
    text = uaroet_assignment_report(decision)
    report = json.loads(text, parse_constant=lambda x: pytest.fail(f"nonstandard JSON {x}"))
    assert report["posterior"]["utility_split_rhat"] == [None, None]
    assert report["posterior"]["utility_rhat_status"] == ["unavailable", "unavailable"]
    assert report["assigned"] == [1, 2] and report["likelihood_used"] == [1, 1]


def test_invalid_callback_rejected_before_randomness():
    rng = np.random.default_rng(2)
    before = rng.bit_generator.state
    with pytest.raises(ValueError, match="on_chunk"):
        fit_uaroet_desktop(study(), rng=rng, on_chunk=3)
    assert rng.bit_generator.state == before
