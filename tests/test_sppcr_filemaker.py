"""Native numeric-export agreement and strict field/identity validation."""

import json
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import (
    format_sppcr_filemaker,
    parse_sppcr_filemaker,
    sppcr_bootstrap,
    sppcr_bootstrap_intervals,
    sppcr_data,
)

FIXTURE = json.loads((Path(__file__).parent / "fixtures/sppcr_filemaker.json").read_text())
BASE = FIXTURE["cases"]["basic"]["input"]


def native(case, tag):
    return np.array(
        [
            [float(x) for x in line.split()[1:]]
            for line in case["stdout"].splitlines()
            if line.split() and line.split()[0] == tag
        ]
    )


@pytest.mark.parametrize("name", list(FIXTURE["cases"]))
def test_native_reader_agreement(name):
    case = FIXTURE["cases"][name]
    assert case["returncode"] == 0
    d = parse_sppcr_filemaker(case["input"])
    np.testing.assert_array_equal(d.dna, native(case, "dna")[0])
    np.testing.assert_array_equal(d.wells, native(case, "wells")[0])
    np.testing.assert_array_equal(d.seen, native(case, "seen"))
    np.testing.assert_array_equal(d.allele_sizes, native(case, "sizes")[0])
    np.testing.assert_array_equal(np.array(d.progenitor) + 1, native(case, "parents")[0])


@pytest.mark.parametrize("name", list(FIXTURE["cases"]))
def test_canonical_roundtrip_uses_original_genome_units(name):
    d = parse_sppcr_filemaker(FIXTURE["cases"][name]["input"])
    encoded = format_sppcr_filemaker(d)
    assert len(encoded.splitlines()) == len(d.dna)
    assert max(map(len, encoded.splitlines())) <= 500
    again = parse_sppcr_filemaker(encoded)
    np.testing.assert_array_equal(d.genome_dna, again.genome_dna)
    np.testing.assert_array_equal(d.dna, again.dna)
    np.testing.assert_array_equal(d.seen, again.seen)
    assert d.progenitor == again.progenitor
    assert d.allele_sizes == again.allele_sizes


def test_explicit_whitespace_extensions_and_quoted_numbers():
    text = "\n# comment\n" + BASE.replace(" ", "\t") + "\n"
    np.testing.assert_array_equal(
        parse_sppcr_filemaker(text).seen, parse_sppcr_filemaker(BASE).seen
    )
    quoted = "\n".join(" ".join('"' + v + '"' for v in row.split()) for row in BASE.splitlines())
    np.testing.assert_array_equal(
        parse_sppcr_filemaker(quoted).seen, parse_sppcr_filemaker(BASE).seen
    )


def test_retained_unseen_progenitor_and_drop_disclosure():
    text = BASE.replace(".5 100 4", ".5 100 0").replace("1 100 10", "1 100 0")
    with pytest.raises(ValueError, match="unseen progenitor"):
        parse_sppcr_filemaker(text)
    d = parse_sppcr_filemaker(text, unseen_alleles="retain")
    assert d.progenitor == (0, 1)
    assert np.all(d.seen[:, 0] == 0)
    d = parse_sppcr_filemaker(FIXTURE["cases"]["drop_unseen"]["input"])
    assert d.omitted_allele_sizes == (104,)


def test_filemaker_to_bootstrap_intervals():
    d = parse_sppcr_filemaker(BASE)
    b = sppcr_bootstrap(
        d.dna,
        d.seen,
        d.wells,
        progenitor=d.progenitor,
        rng=np.random.default_rng(82),
        replicates=20,
    )
    r = sppcr_bootstrap_intervals(b)
    assert np.all(r.frequency.available)
    assert r.progenitor == (0, 1)
    assert not d.seen.flags.writeable


@pytest.mark.parametrize(
    "text",
    [
        "",
        "# comment",
        BASE.replace("100 102 30", "102 100 30"),
        BASE.replace("100 10 102 15", "102 10 100 15"),
        BASE.replace("104 5", "106 5"),
        BASE.replace("104 5", "104"),
        BASE.replace("104 5", ""),
        BASE.replace("100 4", "100 -4"),
        BASE.replace("100 4", "100 40"),
        BASE.replace("20 .5", "20 -.5"),
        BASE.replace("20 .5", "20 1e-999"),
        BASE.replace("20 .5", "20 1e999"),
        BASE.replace("20 .5", "0 .5"),
        BASE.replace("20 .5", "20.0 .5"),
        BASE.replace("20 .5", "2147483648 .5"),
        BASE.replace("104 2", "100 2"),
        BASE.replace("100 102 20", "100 108 20"),
        "100,102,20,.5,100,4,102,8,104,2,",
        "100,,102,20,.5,100,4,102,8,104,2",
        '100,102,20,.5,100,"",102,8,104,2',
        '100,102,20,.5,100,"4 5",102,8,104,2',
        '"100,102,20,.5,100,4,102,8,104,2',
        '100,102,20,.5,100,"4"oops,102,8,104,2',
        BASE.replace("100 4", "100 é4"),
        BASE.replace("100 4", "100 4;"),
    ],
)
def test_malformed_fields_and_inconsistent_rows_rejected(text):
    with pytest.raises(ValueError):
        parse_sppcr_filemaker(text)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(max_runs=1),
        dict(max_alleles=2),
        dict(max_characters=10),
        dict(max_runs=True),
        dict(max_alleles=0),
    ],
)
def test_resource_limits(kwargs):
    with pytest.raises(ValueError):
        parse_sppcr_filemaker(BASE, **kwargs)


def test_formatter_enforces_native_dimensions_integer_and_line_limits():
    for data in [
        sppcr_data([1], np.ones((1, 26)), 10, range(100, 126), (100, 102)),
        sppcr_data(np.ones(51), np.ones((51, 1)), 10, [100], (100, 100)),
        sppcr_data([1], [[2**31]], 2**31, [100], (100, 100)),
        sppcr_data(
            [1],
            np.ones((1, 25)) * 1000000000,
            2000000000,
            range(1000000000, 1000000025),
            (1000000000, 1000000001),
        ),
    ]:
        with pytest.raises(ValueError):
            format_sppcr_filemaker(data)


def test_python_limits_can_be_increased_explicitly():
    row = "100 100 10 1 100 2\n"
    with pytest.raises(ValueError, match="max_runs"):
        parse_sppcr_filemaker(row * 51)
    assert parse_sppcr_filemaker(row * 51, max_runs=51).seen.shape == (51, 1)
