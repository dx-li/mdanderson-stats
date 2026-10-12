"""Compare GUI-valid original desktop conduct boundaries with the Python core."""

import csv
import json
import sys
from pathlib import Path

from mdanderson_stats import BOINDesign

rows = list(csv.DictReader(Path(sys.argv[1]).open()))
checked = 0
excluded = 0
for row in rows:
    target = float(row["target"])
    runin = row["runin"] == "TRUE"
    # The original GUI only offers this option at target .25/cohort size 3.
    # Keep out-of-GUI original probes separate from supported Python settings.
    if runin and target != 0.25:
        excluded += 1
        continue
    table = BOINDesign(
        target=target, extra_safe=row["extra_safe"] == "TRUE", stay_at_one_of_three=runin
    ).boundary_table(18)
    index = int(row["n"]) - 1
    for name, values in (
        ("escalate", table.escalate_max),
        ("deescalate", table.deescalate_min),
        ("eliminate", table.eliminate_min),
    ):
        expected = int(row[name]) if row[name] else index + 2
        if expected != values[index]:
            raise AssertionError((row, name, values[index]))
    checked += 1
assert checked == 144 and excluded == 72
print(
    json.dumps(
        {
            "matched_rows": checked,
            "matched_values": checked * 3,
            "outside_gui_rows_not_compared": excluded,
        }
    )
)
