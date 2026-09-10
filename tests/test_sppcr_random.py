"""Independent native SPPCR random-stream reconciliation with existing RANDLIB."""

import json
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import RandlibGenerator

FIXTURE = json.loads((Path(__file__).parent / "fixtures/sppcr_random.json").read_text())


def native_rows(case):
    return [
        (words[0], [float(v) for v in words[1:]])
        for line in case["stdout"].splitlines()
        if (words := line.split())
    ]


def replay(commands):
    g = RandlibGenerator()
    rows = []
    lines = iter(commands)
    for line in lines:
        words = line.split()
        op = words[0]
        if op == "select":
            g.select(int(words[1]))
        elif op == "all":
            g.set_all_seeds(tuple(map(int, words[1:])))
        elif op == "seed":
            g.set_seeds(tuple(map(int, words[1:])))
        elif op == "anti":
            g.set_antithetic(words[1] == "1")
        elif op == "reset":
            g.reinitialize(int(words[1]))
        elif op == "advance":
            g.advance_state(int(words[1]))
        elif op == "phrase":
            pair = g.set_phrase(next(lines), source="fortran")
            rows.append(("phrase", list(pair)))
        elif op == "integer":
            rows.append(("integer", g.integers(int(words[1])).tolist()))
        elif op == "uniform":
            rows.append(
                ("uniform", g.uniform(int(words[1]), legacy=True, source="fortran").tolist())
            )
        elif op == "binomial":
            rows.append(
                (
                    "binomial",
                    g.binomial(
                        int(words[3]),
                        n=int(words[1]),
                        p=float(words[2]),
                        legacy=True,
                        source="fortran",
                    ).tolist(),
                )
            )
        else:
            raise AssertionError(op)
        rows.append(("state", [g.stream, *g.get_seeds()]))
    return rows


@pytest.mark.parametrize("name", [n for n in FIXTURE["cases"] if n != "reseed_other_stream_defect"])
def test_source_draws_and_every_recorded_state(name):
    case = FIXTURE["cases"][name]
    expected = native_rows(case)
    actual = replay(case["commands"])
    assert len(actual) == len(expected)
    for (tag, a), (native_tag, b) in zip(actual, expected, strict=True):
        assert tag == native_tag
        np.testing.assert_array_equal(a, b)


def test_native_reseed_other_stream_leaves_stream_one_stale_python_repairs_it():
    case = FIXTURE["cases"]["reseed_other_stream_defect"]
    expected = native_rows(case)
    actual = replay(case["commands"])
    # The fourth command selects stream 1 after reseeding from stream 7.
    assert actual[-3] == ("state", [1, 111, 222])
    assert expected[-3] != actual[-3]
    assert actual[-2][0] == "integer"
    a, b = 111, 222
    independently = []
    for _ in range(10):
        a = a * 40014 % 2147483563
        b = b * 40692 % 2147483399
        z = a - b
        independently.append(z if z >= 1 else z + 2147483562)
    np.testing.assert_array_equal(actual[-2][1], independently)
    assert actual[-1] == ("state", [1, a, b])


def test_legacy_probability_rounding_safeguard_preserves_state():
    g = RandlibGenerator()
    before = g.get_seeds()
    with pytest.raises(ValueError, match="endpoint"):
        g.binomial(2, n=100, p=1e-100, legacy=True, source="fortran")
    assert g.get_seeds() == before


def test_legacy_sppcr_generation_matches_native_cell_order_and_final_state():
    from mdanderson_stats import sppcr_generate_legacy

    g = RandlibGenerator()
    p = [[0.1, 0.4, 0.8], [0.2, 0.5, 0.9]]
    samples = sppcr_generate_legacy(p, [20, 40], rng=g, replicates=3)
    rows = native_rows(FIXTURE["cases"]["generation_grid"])
    values = [row[1][0] for row in rows if row[0] == "binomial"]
    np.testing.assert_array_equal(samples.seen, np.array(values).reshape(3, 2, 3))
    np.testing.assert_array_equal(samples.probability, np.array(p, dtype=np.float32).astype(float))
    assert list(g.get_seeds()) == rows[-1][1][1:]
    assert not samples.seen.flags.writeable
    np.testing.assert_array_equal(
        samples.seen + samples.unseen,
        np.broadcast_to(np.array([20, 40])[:, None], samples.seen.shape),
    )


@pytest.mark.parametrize(
    "p,n,replicates", [(1e-100, 10, 1), (1 - 1e-10, 10, 1), (0.5, 2147483647, 1), (0.5, 10, -1)]
)
def test_legacy_generation_rejects_entire_invalid_design_before_drawing(p, n, replicates):
    from mdanderson_stats import sppcr_generate_legacy

    g = RandlibGenerator()
    before = g.get_seeds()
    with pytest.raises(ValueError):
        sppcr_generate_legacy([[0.5, p]], n, rng=g, replicates=replicates)
    assert g.get_seeds() == before


def test_legacy_zero_replicates_and_endpoints():
    from mdanderson_stats import sppcr_generate_legacy

    g = RandlibGenerator()
    before = g.get_seeds()
    empty = sppcr_generate_legacy([[0, 1]], 10, rng=g, replicates=0)
    assert empty.seen.shape == (0, 1, 2) and g.get_seeds() == before
    sample = sppcr_generate_legacy([[0, 1]], 10, rng=g, replicates=2)
    np.testing.assert_array_equal(sample.seen, [[[0, 10]], [[0, 10]]])
