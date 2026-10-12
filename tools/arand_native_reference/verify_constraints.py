"""300 seeded original-instruction single-group constraint comparisons.

Set ARAND_REFERENCE_LIBRARY; requires installed mdanderson-stats plus the
reference harness dependencies. Trial limits are not reached in these probes.
"""

import random

import arand_minmax_reference as mm

from mdanderson_stats.arand_native import ArandNativeArmLimits, arand_native_constraints

rng = random.Random(314)
for _ in range(300):
    k = rng.randint(1, 4)
    statuses = [rng.choice([0, 1, 2, 3, 4, 5, 9]) for _ in range(k)]
    n = [rng.randint(0, 8) for _ in range(k)]
    lo = [rng.randint(0, 8) for _ in range(k)]
    hi = [rng.randint(0, 8) for _ in range(k)]
    gl = [rng.randint(0, 8) for _ in range(k)]
    gh = [rng.randint(0, 8) for _ in range(k)]
    start = [rng.choice([0, 5, 10]) for _ in range(k)]
    dur = [rng.choice([0, 5]) for _ in range(k)]
    now = rng.choice([0, 5, 10, 15, 20])
    cfg = dict(
        statuses=statuses,
        now=now,
        arm_count=n,
        arm_min=lo,
        arm_max=hi,
        group_count=n,
        group_min=gl,
        group_max=gh,
        start=start,
        duration=dur,
    )
    expected = mm.run(**cfg)["statuses"]
    limits = tuple(ArandNativeArmLimits(*a) for a in zip(lo, hi, gl, gh, start, dur))
    actual = arand_native_constraints(tuple(statuses), tuple(n), limits, now=now)
    assert tuple(expected) == actual, (cfg, expected, actual)
print("300 original ordered arm/group constraint comparisons passed")
