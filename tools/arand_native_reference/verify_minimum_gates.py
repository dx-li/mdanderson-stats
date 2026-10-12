import random

import arand_mincriteria_reference as ref

from mdanderson_stats.arand_native import ArandNativeGates

r = random.Random(42)
for j in range(500):
    k = 2
    n = [r.randrange(6) for _ in range(k)]
    lo = [r.randrange(6) for _ in range(k)]
    gl = [r.randrange(6) for _ in range(k)]
    now = r.randrange(6)
    mi = r.randrange(6)
    mt = r.randrange(6)
    g = ArandNativeGates(*(bool(r.randrange(2)) for _ in range(8)))
    expected = ref.run(
        [0, 0],
        now=now,
        count=sum(n),
        min_count=mi,
        min_time=mt,
        arm_count=n,
        group_count=n,
        arm_min=lo,
        group_min=gl,
        gates=list(vars(g).values()),
    )
    trial = (not g.trial_minimum or sum(n) >= mi) and (not g.duration_minimum or now >= mt)
    am = any(a >= b for a, b in zip(n, lo))
    gm = any(a >= b for a, b in zip(n, gl))
    actual = trial and (
        (am or gm)
        if g.arm_minimum and g.group_minimum
        else am
        if g.arm_minimum
        else gm
        if g.group_minimum
        else True
    )
    assert actual == expected, (j, n, lo, gl, now, mi, mt, g, actual, expected)
print("500 minimum gate comparisons passed")
