"""Reopen a PRT history and export a qualified conduct report.

The six numeric observations are the bundled conduct-guide example. Holds are
expected when a profile cannot support a stable, valid computation.
"""

from dataclasses import replace

import numpy as np

from mdanderson_stats import PRTPatient, PRTStudy, analyze_prt_conduct, read_prt_study

study = PRTStudy(
    intervals=6,
    doses=6,
    patients=tuple(
        PRTPatient(dose, interval, True)
        for dose, interval in [(1, 6), (1, 5), (1, 5), (2, 2), (2, 1), (2, 1)]
    ),
)
reopened = read_prt_study(study.to_json())
# Advance the last patient's observed follow-up by one interval and save again.
updated = replace(reopened, patients=(*reopened.patients[:-1], PRTPatient(2, 2, True)))
for profile in ("paper_2008", "legacy_1_0"):
    result = analyze_prt_conduct(
        read_prt_study(updated.to_json()),
        rng=np.random.default_rng(20261012),
        profile=profile,
    )
    print(result.to_json())
    print(result.to_csv())
