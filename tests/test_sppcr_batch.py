"""Native parser agreement, validated identities, units and canonical round trips."""

import json
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import (
    format_sppcr_batch,
    parse_sppcr_batch,
    sppcr_bootstrap,
    sppcr_bootstrap_intervals,
    sppcr_data,
)

FIXTURE = json.loads((Path(__file__).parent / "fixtures/sppcr_batch.json").read_text())
BASE = FIXTURE["cases"]["basic"]["input"]


def native_rows(case, tag):
    return np.array(
        [
            [float(x) for x in line.split()[1:]]
            for line in case["stdout"].splitlines()
            if line.split() and line.split()[0] == tag
        ]
    )


@pytest.mark.parametrize("name", ["basic", "continued_comments", "exponent_dna", "drop_unseen"])
def test_valid_native_parser_agreement(name):
    case = FIXTURE["cases"][name]
    assert case["returncode"] == 0
    d = parse_sppcr_batch(case["input"])
    for actual, tag in [(d.dna, "dna"), (d.wells, "wells"), (d.allele_sizes, "sizes")]:
        np.testing.assert_array_equal(actual, native_rows(case, tag)[0])
    np.testing.assert_array_equal(d.seen, native_rows(case, "seen"))
    np.testing.assert_array_equal(np.array(d.progenitor) + 1, native_rows(case, "parents")[0])
    np.testing.assert_array_equal(2 * d.genome_dna, d.dna)


def test_unseen_progenitor_native_invalid_index_and_explicit_retain_policy():
    case = FIXTURE["cases"]["unseen_parent"]
    assert native_rows(case, "parents")[0, 0] == -1
    with pytest.raises(ValueError, match="unseen progenitor"):
        parse_sppcr_batch(case["input"])
    d = parse_sppcr_batch(case["input"], unseen_alleles="retain")
    assert d.progenitor == (0, 1)
    assert d.allele_sizes == (100, 102, 104)
    assert np.all(d.seen[:, 0] == 0)


def test_unused_native_storage_does_not_change_python_columns():
    case = FIXTURE["cases"]["stale_unused_row"]
    assert native_rows(case, "dimensions")[0, 1] == 3
    d = parse_sppcr_batch(case["input"])
    assert d.allele_sizes == (100, 102)
    assert d.omitted_allele_sizes == (104,)
    parse_sppcr_batch(BASE)
    assert parse_sppcr_batch(case["input"]).omitted_allele_sizes == (104,)


def test_native_negative_sign_loss_is_rejected():
    case = FIXTURE["cases"]["negative_seen"]
    assert "-4" in case["input"] and native_rows(case, "seen")[0, 0] == 4
    with pytest.raises(ValueError):
        parse_sppcr_batch(case["input"])


@pytest.mark.parametrize("name", ["basic", "continued_comments", "exponent_dna", "drop_unseen"])
def test_canonical_roundtrip_and_dna_conversion_once(name):
    d = parse_sppcr_batch(FIXTURE["cases"][name]["input"])
    encoded = format_sppcr_batch(d)
    again = parse_sppcr_batch(encoded)
    assert max(map(len, encoded.splitlines())) <= 240
    np.testing.assert_array_equal(d.genome_dna, again.genome_dna)
    np.testing.assert_array_equal(d.dna, again.dna)
    np.testing.assert_array_equal(d.seen, again.seen)
    assert d.allele_sizes == again.allele_sizes
    assert d.progenitor == again.progenitor


def test_long_integer_records_wrap_and_round_trip():
    d = sppcr_data(
        [0.25], np.ones((1, 50)), 10, list(range(10000000, 10000050)), (10000001, 10000002)
    )
    text = format_sppcr_batch(d)
    assert max(map(len, text.splitlines())) <= 240
    again = parse_sppcr_batch(text)
    assert d.allele_sizes == again.allele_sizes
    np.testing.assert_array_equal(d.seen, again.seen)


def test_tabs_accepted_as_explicit_whitespace_extension():
    np.testing.assert_array_equal(
        parse_sppcr_batch(BASE.replace(" ", "\t")).seen, parse_sppcr_batch(BASE).seen
    )


def test_parsed_data_to_bootstrap_intervals():
    d = parse_sppcr_batch(BASE)
    b = sppcr_bootstrap(
        d.dna,
        d.seen,
        d.wells,
        progenitor=d.progenitor,
        rng=np.random.default_rng(22),
        replicates=10,
    )
    limits = sppcr_bootstrap_intervals(b)
    assert np.all(limits.frequency.available)
    assert limits.progenitor == (0, 1)


@pytest.mark.parametrize(
    "text",
    [
        "",
        BASE.replace("nallele 3", "nallele 0"),
        BASE.replace("nrun 2", "nrun 0"),
        BASE.replace("nallele 3", "nallele 51"),
        BASE.replace("nwell 20 30", "nwell 20"),
        BASE.replace("nwell 20 30", "nwell 20 30 40"),
        BASE.replace("nwell 20 30", "nwell 20.0 30"),
        BASE.replace("nwell 20 30", "nwell 0 30"),
        BASE.replace("4 8 2", "40 8 2"),
        BASE.replace("100 102 104", "100 100 104"),
        BASE.replace("progenitor 100 102", "progenitor 100 106"),
        BASE.replace("run 0.5", "run\n0.5"),
        BASE.replace("run 0.5", "run -0.5"),
        BASE.replace("run 0.5", "run 1e-999"),
        BASE.replace("run 0.5", "run 1e999"),
        BASE.replace("nwell 20 30", "nwell 2147483648 30"),
        BASE + "run 1 1 1 1\n",
        BASE.replace("nallele 3", "nallele = 3"),
        BASE.replace("4 8 2", "4,8,2"),
        BASE.replace("nrun 2\n", ""),
        BASE.rsplit("run 1", 1)[0],
    ],
)
def test_malformed_or_inconsistent_data_rejected(text):
    with pytest.raises(ValueError):
        parse_sppcr_batch(text)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(max_runs=1),
        dict(max_characters=10),
        dict(max_alleles=0),
        dict(max_runs=True),
        dict(unseen_alleles="ignore"),
    ],
)
def test_explicit_limits(kwargs):
    with pytest.raises(ValueError):
        parse_sppcr_batch(BASE, **kwargs)


def test_data_factory_ownership_homozygote_and_all_zero_retention():
    x = np.zeros((1, 2))
    d = sppcr_data([0.5], x, 10, [100, 102], (100, 100), unseen_alleles="retain")
    x[:] = 1
    assert d.progenitor == (0, 0)
    assert not np.any(d.seen)
    for value in [d.genome_dna, d.dna, d.wells, d.seen]:
        with pytest.raises(ValueError):
            value.setflags(write=True)
    with pytest.raises(ArithmeticError):
        sppcr_data([1e308], [[1]], 10, [100], (100, 100))
    large = sppcr_data([1], [[2**31]], 2**31, [100], (100, 100))
    with pytest.raises(ValueError, match="int32"):
        format_sppcr_batch(large)
