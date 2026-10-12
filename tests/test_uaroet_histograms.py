"""Native bin boundaries, complete chunk coverage and distributions of summed risks."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import (
    UAROETDesktopStudy,
    UAROETPosteriorHistograms,
    fit_uaroet_desktop,
    read_uaroet_desktop_study,
)

ROOT = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def inputs():
    study = replace(
        read_uaroet_desktop_study((ROOT / "uaroet-python-study.json").read_text()),
        patients=(),
        simulation_parameters={"NumChains": 1, "BurnInSamples": 0},
    )
    chunks = []
    fit_uaroet_desktop(
        study,
        rng=np.random.default_rng(713),
        target=0.99,
        chunk_draws=8,
        max_draws=8,
        on_chunk=lambda c, n, f: chunks.append(f),
    )
    return study, chunks[0]


@pytest.mark.parametrize(
    "case", json.loads((ROOT / "uaroet-desktop-histogram.json").read_text())["cases"]
)
def test_native_histogram_boundaries(case, inputs):
    s, chunk = inputs
    p = case["probability"]
    joint = np.broadcast_to(np.array([[p, 0], [1 - p, 0]]), (1, 1, 2, 2, 2))
    h = UAROETPosteriorHistograms(s, profile="legacy_1_8")
    h(0, 0, replace(chunk, joint=joint))
    h.validate_complete(chains=1, draws_per_chain=1)
    assert h.counts[0, case["bin"], 0] == 1
    safe = UAROETPosteriorHistograms(s)
    safe(0, 0, replace(chunk, joint=joint))
    assert safe.counts[0, min(99, int(np.floor(p * 100))), 0] == 1


def test_summed_risk_histograms_are_not_summed_category_histograms(inputs):
    base, chunk = inputs
    study = UAROETDesktopStudy(
        2,
        4,
        3,
        np.zeros(10),
        np.ones(10),
        100 * np.outer(np.arange(4) / 3, 1 - np.arange(3) / 2),
        dict(base.trial_parameters, LowestAcceptableEffLevel=2),
        base.simulation_parameters,
    )
    joint = np.broadcast_to(np.outer([0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.5]), (1, 1, 2, 4, 3))
    h = UAROETPosteriorHistograms(study)
    h(0, 0, replace(chunk, joint=joint))
    for name, index in [("PEff", 70), ("PTox", 80), ("PUtil", 23)]:
        assert h.counts[0, index, h.quantities.index(name)] == 1
    assert h.counts[0, 70, h.quantities.index("PE2")] == 0
    assert h.counts[0, 70, h.quantities.index("PE3")] == 0
    assert len(h.to_csv(0).splitlines()) == 101
    assert not h.counts.flags.writeable


def test_actual_streamed_histograms_match_independent_numpy_histogram(inputs):
    s, _ = inputs
    s = replace(s, simulation_parameters={"NumChains": 2, "BurnInSamples": 0})
    h = UAROETPosteriorHistograms(s)
    chunks = []

    def callback(c, n, f):
        h(c, n, f)
        chunks.append(f.joint[0])

    fit = fit_uaroet_desktop(
        s,
        rng=np.random.default_rng(4),
        target=1e-6,
        chunk_draws=16,
        max_draws=32,
        on_chunk=callback,
    )
    h.validate_complete(chains=2, draws_per_chain=fit.draws_per_chain)
    j = np.concatenate(chunks)
    expected, _ = np.histogram(j[:, 0].sum(-1)[:, 0] * 100, bins=np.arange(101))
    np.testing.assert_array_equal(h.counts[0, :, 0], expected)
    with pytest.raises(ValueError, match="coverage"):
        h.validate_complete(chains=1, draws_per_chain=32)


def test_duplicate_or_invalid_chunks_do_not_mutate_histograms(inputs):
    s, chunk = inputs
    h = UAROETPosteriorHistograms(s)
    h(0, 0, chunk)
    before = h.counts
    with pytest.raises(ValueError, match="contiguous"):
        h(0, 0, chunk)
    with pytest.raises(ArithmeticError):
        h(0, 8, replace(chunk, joint=chunk.joint * 2))
    np.testing.assert_array_equal(h.counts, before)
