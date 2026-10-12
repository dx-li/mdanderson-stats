"""Installed-package smoke for all original defaults; originals remain external.

The small simulations exercise workflows, not clinical precision or validation.
Usage: python this_script EXTRACTED_DATAFILES_DIRECTORY
"""

import sys
from pathlib import Path

import numpy as np

import mdanderson_stats as m

print("Imported package:", m.__file__, flush=True)
root = Path(sys.argv[1])
for path in sorted(root.glob("Default*.html")):
    study = m.read_boin_desktop_project(path.read_text(encoding="utf-8-sig"))
    assert m.read_boin_desktop_project(study.to_html()) == study
    if study.mode in ("single", "tite"):
        for language in ("en", "zh"):
            assert m.boin_desktop_univariate_protocol(study, language=language)
        for profile in ("safe", "desktop_legacy"):
            workflow = m.simulate_boin_desktop_workflow(
                study, batches=2, trials=3, seed=712, profile=profile
            )
            report = m.boin_desktop_workflow_html(workflow)
            assert m.read_boin_desktop_project(report) == study
            print(path.name, profile, "saved simulation/comparison/report passed", flush=True)
        study = m.set_boin_desktop_single_counts(study, [3, 3, 12, 6, 0], [0, 0, 3, 3, 0])
        selected = m.select_boin_desktop_single(study, outcomes_complete=True)
        assert selected.dose is not None
        print(path.name, study.mode, "saved final selection", selected.dose, flush=True)
    else:
        for profile in ("safe", "desktop_legacy"):
            batch = m.simulate_boin_desktop_scenarios(
                study, batches=2, trials=3, seed=712, profile=profile
            )
            assert len(batch.results) == len(study.scenarios) * 2
            report = m.boin_desktop_simulation_html(batch)
            assert m.read_boin_desktop_project(report) == study
            for result in batch.results:
                if profile == "safe":
                    np.testing.assert_array_equal(result.patients, result.actual_patients)
                    assert not result.selected_eliminated.any()
            print(
                path.name, profile, len(batch.results), "scenario batches/report passed", flush=True
            )
        for language in ("en", "zh"):
            assert m.boin_desktop_combination_protocol(study, language=language)
        patient_study = m.add_boin_desktop_patient(
            study, patient_id=1, date_entered="2026-10-12", seed=17
        )
        result = m.recommend_boin_desktop(patient_study, seed=17)
        assert (
            m.read_boin_desktop_project(m.boin_desktop_conduct_html(patient_study, result))
            == patient_study
        )
        print(path.name, "bilingual methods and patient conduct report passed", flush=True)
