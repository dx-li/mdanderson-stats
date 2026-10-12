"""Installed current-EffTox saved-file, conduct and all-scenario smoke exercise.

Usage: python check_efftox_desktop_workflows.py STUDY_XML OUTPUT_DIRECTORY
Uses explicitly small MCMC settings; this is workflow evidence, not a mixing test.
"""

import hashlib
import json
import sys
from pathlib import Path

import mdanderson_stats
from mdanderson_stats.efftox_desktop import read_efftox_desktop_xml
from mdanderson_stats.efftox_desktop_batches import (
    efftox_desktop_batches_html,
    simulate_efftox_desktop_scenarios,
    summarize_efftox_desktop_batches,
)
from mdanderson_stats.efftox_desktop_controller import recommend_efftox_desktop
from mdanderson_stats.efftox_desktop_inference import EffToxDesktopInference
from mdanderson_stats.efftox_desktop_patients import (
    add_efftox_desktop_patient,
    clear_efftox_desktop_patients,
    delete_last_efftox_desktop_patient,
    update_efftox_desktop_patient,
)
from mdanderson_stats.efftox_desktop_report import efftox_desktop_conduct_html

source, output = Path(sys.argv[1]), Path(sys.argv[2])
output.mkdir(parents=True, exist_ok=True)
raw = source.read_bytes()
study = read_efftox_desktop_xml(raw.decode("latin1"))
assert not study.patients
assert "site-packages" in str(mdanderson_stats.__file__)


def inference():
    return EffToxDesktopInference(
        study, monotone_toxicity=False, seed=730, draws=32, warmup=16, chains=2
    )


batches = []
for seed in (130, 230):
    service = inference()
    batches.extend(simulate_efftox_desktop_scenarios(study, service, trials=1, seed=seed))
report = efftox_desktop_batches_html(batches)
(output / "all-scenarios.html").write_text(report, encoding="utf-8")
scenarios = []
for index in range(len(study.scenarios)):
    group = [b for b in batches if b.scenario_index == index]
    summary = summarize_efftox_desktop_batches(group)
    assert summary.trials == 2
    assert all(b.counts.sum() <= study.max_patients for b in group)
    scenarios.append(
        dict(
            scenario=index,
            seeds=[b.seed for b in group],
            patients=[int(b.counts.sum()) for b in group],
            selected=[int(b.selected_dose_index[0]) for b in group],
            rselect=summary.rselect,
        )
    )
service = inference()
history = study
for identifier in range(1, study.cohort_size + 1):
    history = add_efftox_desktop_patient(
        history, service, patient_id=identifier, begin_date=f"literal day {identifier}"
    )
pending = recommend_efftox_desktop(history, service)
(output / "pending-conduct.html").write_text(
    efftox_desktop_conduct_html(history, pending, inference=service), encoding="utf-8"
)
for patient in history.patients:
    history = update_efftox_desktop_patient(history, patient.patient_id, outcome=0)
assert history.observed_counts()[:, 0, 1].sum() == study.cohort_size
history = update_efftox_desktop_patient(
    history, 1, new_patient_id=1001, begin_date="literal <corrected> date"
)
assert read_efftox_desktop_xml(history.to_xml()) == history
history = delete_last_efftox_desktop_patient(history)
assert len(history.patients) == study.cohort_size - 1
assert not clear_efftox_desktop_patients(history).patients
record = dict(
    package=mdanderson_stats.__file__,
    source_sha256=hashlib.sha256(raw).hexdigest(),
    workflow_smoke_only=True,
    draws=32,
    warmup=16,
    chains=2,
    scenarios=scenarios,
    pending_status=pending.status,
    pending_possible_completions=pending.completion_count,
    pending_analyses=pending.analyses,
    patient_edit_delete_clear_roundtrip=True,
    reports_created=True,
)
(output / "validation.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
