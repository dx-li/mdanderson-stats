import itertools
import json
from pathlib import Path

import arand_mincriteria_reference as mi
import arand_minmax_reference as mm

out = []
for flags in itertools.product([False, True], repeat=8):
    cfg = dict(
        statuses=[1, 3],
        now=5,
        count=5,
        min_count=6,
        max_count=8,
        min_time=6,
        max_time=9,
        arm_count=[1, 4],
        group_count=[1, 4],
        arm_min=[2, 3],
        group_min=[0, 5],
        arm_max=[1, 8],
        group_max=[8, 4],
        gates=flags,
    )
    out.append(dict(input=cfg, minimum=mi.run(**cfg), constraints=mm.run(**cfg)))
Path("research/fixtures/arand-native/constraint-gates.json").write_text(
    json.dumps(out, indent=2) + "\n"
)
