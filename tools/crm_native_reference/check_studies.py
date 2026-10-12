"""Smoke original installer studies against the imported (optionally installed) package.

Usage: python check_studies.py /path/to/Suite/examples /path/to/older/examples
Low DA draw counts verify plumbing only, not Monte Carlo adequacy.
"""

import sys
from datetime import date
from pathlib import Path

import mdanderson_stats as m

print("Package:", m.__file__, flush=True)
for directory in sys.argv[1:]:
    for path in sorted(Path(directory).glob("*.html")):
        text = path.read_text(encoding="utf-8-sig")
        study = m.read_crm_desktop_html(text)
        run = m.simulate_crm_desktop_study(study, trials=1, seed=2026, draws=8, warmup=0, chains=2)
        report = run.to_html()
        assert m.read_crm_desktop_html(report).skeletons == study.skeletons
        assert "Active prior-median skeletons" in report
        history = m.read_crm_desktop_history(text, date_order="mdy")
        first = m.recommend_crm_desktop(history, at=date(2026, 10, 12), seed=2026)
        assert first.can_add_patient
        added = first.add_patient(patient_id=101, fingerprint=101)
        assert (
            m.read_crm_desktop_history(added.to_html(), date_order="mdy").patients == added.patients
        )
        print(path.name, len(run.simulations), "scenarios; report/history round-trip", flush=True)
