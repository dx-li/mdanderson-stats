import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats import SOGS_MOUSE_LENGTHS, sogs_eligible, sogs_recombine, sogs_screen


def test_compiled_native_recombination_and_screening():
    cases = json.loads((Path(__file__).parent / "fixtures/sogs-native-kernels.json").read_text())
    for c in cases:
        result = sogs_recombine(c["segments"], c["length"], c["breakpoints"])
        for j, part in enumerate(("retained", "transferred")):
            segments = getattr(result, part)
            assert_allclose(segments, np.asarray(c[part]).reshape(-1, 2), rtol=0, atol=1e-5)
            scan = sogs_screen(segments, c["length"])
            assert scan.missed == c["missed"][j]
            assert_allclose(scan.donor_length, c["donor_length"][j], rtol=0, atol=1e-5)
            assert not segments.flags.writeable


def test_selection_rules_match_native_first_eligible_and_retain_ties():
    lengths = [1, 3, 7, 20]
    pure = np.array([[1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 1, 0], [1, 0, 1, 0]], dtype=bool)
    # Original one_simulation, fixed candidates, IGN UIN stub selects first eligible:
    # rule, true IPT count, donor length = (0,2,27),(1,2,27),(2,2,23),(3,1,11).
    for rule, indices, count, donor_length in [
        (0, [0, 1, 2, 3], 2, 27),
        (1, [0, 2, 3], 2, 27),
        (2, [2, 3], 2, 23),
        (3, [1], 1, 11),
    ]:
        eligible = sogs_eligible(pure, lengths, rule=rule)
        assert_array_equal(eligible, indices)
        first = pure[eligible[0]]
        assert first.sum() == count
        assert np.sum(np.asarray(lengths)[~first]) == donor_length
        assert not eligible.flags.writeable


def test_conservation_endpoints_and_invalid_genomes():
    rng = np.random.default_rng(56)
    for _ in range(50):
        endpoints = np.sort(rng.uniform(0, 100, 20)).reshape(-1, 2)
        breaks = np.sort(rng.uniform(0, 100, 15))
        out = sogs_recombine(endpoints, 100, breaks)
        got = (
            sogs_screen(out.retained, 100).donor_length
            + sogs_screen(out.transferred, 100).donor_length
        )
        assert_allclose(got, np.diff(endpoints, axis=1).sum(), atol=1e-12, rtol=0)
    assert sogs_screen([[5, 5.1]], 20).detected
    assert sogs_screen([[4.9, 5]], 20).detected
    assert sogs_screen([[5.01, 9.99]], 20).missed
    assert sogs_screen([], 20).pure_ipt
    assert not sogs_screen([], 20).missed
    assert_array_equal(sogs_screen([[0.1, 0.2]], 1).probes, [0, 1])
    assert SOGS_MOUSE_LENGTHS.size == 19
    assert not SOGS_MOUSE_LENGTHS.flags.writeable
    with pytest.raises(ValueError, match="separated"):
        sogs_recombine([[0, 5], [4, 9]], 20, [])
