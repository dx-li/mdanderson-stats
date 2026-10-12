"""Installed mTPI workflow/precision check; not a clinical validation study."""

import json
import time
from pathlib import Path

import numpy as np

import mdanderson_stats as m

started = time.monotonic()
study = m.MTPIDesktopStudy(
    0.3,
    0.05,
    0.05,
    30,
    3,
    1,
    ((0.1, 0.2, 0.3, 0.45), (0.25, 0.4, 0.55, 0.7), (0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 1.0)),
    2000,
)
records = []
for profile in ("safe", "desktop_legacy"):
    result = m.simulate_mtpi_desktop(study, profile=profile, seed=72)
    assert np.allclose(result.selection_probability.sum(axis=1), 1)
    assert np.allclose(result.stopping_probability.sum(axis=1), 1)
    assert np.max(result.selection_mcse) <= 0.5 / np.sqrt(study.trials) + 1e-15
    assert np.all(result.mean_total_patients <= result.actual_capacity)
    assert result.selection_probability[-1, -1] == 1
    assert result.stopping_probability[-1, 0] == 1
    assert result.unsafe_assignment_probability[-1] == 0
    saved = m.read_mtpi_desktop_report(m.mtpi_desktop_simulation_report(result))
    assert (saved.study, saved.profile, saved.seed) == (study, profile, 72)
    if profile == "safe":
        assert np.all(result.unsafe_assignment_probability == 0)
        assert result.stopping_probability[-2, 2] == 1
    else:
        assert result.stopping_probability[-2, 1] == 1
    records.append(
        dict(
            profile=profile,
            trials=study.trials * len(study.scenarios),
            max_mcse=float(result.selection_mcse.max()),
            selection=result.selection_probability.tolist(),
            stops=result.stopping_probability.tolist(),
            mean_total=result.mean_total_patients.tolist(),
            unsafe=result.unsafe_assignment_probability.tolist(),
            uniforms=result.uniforms_consumed,
        )
    )
record = dict(
    import_location=m.__file__,
    total_trials=sum(r["trials"] for r in records),
    seconds=time.monotonic() - started,
    records=records,
    scope="Seeded precision, enrollment, safety and report checks; no clinical validity claim",
)
Path("mtpi-desktop-quality.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
