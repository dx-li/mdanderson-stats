"""Deterministic bounded differential checks; set ARAND_REFERENCE_LIBRARY.

Requires the installed Python package and original-reference dependencies.
Run from any directory; the sibling reference imports validate the library hash.
"""

import random

import arand_final_reference as fin
import arand_groupfinal_reference as grp
import arand_status_reference as ref

from mdanderson_stats.arand_native import arand_native_select, arand_native_statuses

rng = random.Random(123)
for i in range(300):
    k = rng.randint(1, 4)
    p = [rng.choice([0, 0.09, 0.1, 0.5, 0.95, 0.96, 1]) for _ in range(k)]
    s = [rng.choice([0, 1, 2, 3, 4, 5, 9]) for _ in range(k)]
    cfg = dict(
        probabilities=p,
        statuses=s,
        lower=rng.choice([0, 0.1]),
        upper=rng.choice([0.95, 1]),
        min_action=rng.choice([1, 3]),
        max_action=rng.choice([1, 2, 3]),
        futility=None if i % 2 else [rng.choice([0.1, 0.2, 0.8]) for _ in range(k)],
    )
    actual = arand_native_statuses(**cfg)
    expected = tuple(ref.run(**cfg)["statuses"])
    assert actual == expected, (cfg, actual, expected)
    for g in (False, True):
        cfg = dict(
            probabilities=p,
            statuses=s,
            cutoff=rng.choice([0.3, 0.5, 1]),
            upper=rng.choice([0.35, 0.95, 1]),
            completed=bool(i % 2),
        )
        actual = arand_native_select(**cfg, group=g)
        expected = (grp if g else fin).run(**cfg)[-1]
        assert str(actual.winner) == expected[2] and actual.early == bool(expected[3]), (
            cfg,
            g,
            actual,
            expected,
        )
print("900 randomized original-reference status/selection comparisons passed")
