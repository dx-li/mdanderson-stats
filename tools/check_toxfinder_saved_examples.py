"""Smoke all five original ToxFinder saved scenarios and patient/report workflows.

Supply the extracted Gem-CTX example directory. Tiny sampler settings test
plumbing only; the resulting reports are not clinical or precision validation.
"""

import shlex
import sys
from pathlib import Path

import mdanderson_stats as m

print("Imported package:", m.__file__, flush=True)
root = Path(sys.argv[1])


def settings(text):
    result = {}
    for line in text.splitlines():
        bits = shlex.split(line, comments=True)
        if bits:
            result[bits[0]] = bits[1:]
    return result


base = settings((root / "init.tmml").read_text())
initial = m.read_toxfinder_tmml((root / "init.tmml").read_text())
elicitation = m.elicit_toxfinder_study_prior(initial, seed=2026)
print(
    "Calibration assessment",
    [
        (a.assessed_tail_probability.tolist(), a.assessed_cutoff_met)
        for a in (elicitation.agent1, elicitation.agent2)
    ],
    flush=True,
)
for scenario in range(5):
    values = base | settings((root / f"curve{scenario}.tmml").read_text())
    text = "\n".join(shlex.join([k, *v]) for k, v in values.items())
    study = m.read_toxfinder_tmml(text)
    run = m.simulate_toxfinder_study(
        study, prior=elicitation, trials=2, seed=2027 + scenario, draws=8, warmup=0
    )
    assert all(t.subjects == 60 and t.posterior_subjects == 60 for t in run.simulation.trials)
    report = run.to_html()
    assert "Independent prior-calibration assessment" in report
    assert m.read_toxfinder_tmml(study.to_tmml()).settings == study.settings
    print(
        "Scenario",
        scenario,
        "2 complete 60-patient trials, final refit/report/TMML passed",
        flush=True,
    )
records = m.read_toxfinder_subject_csv(
    "1,p1,2026,10,12,10:00,1,100,50,0\n2,p2,2026,10,12,11:00,1,100,50,2\n",
    reference_doses=initial.reference_doses,
)
report = m.recommend_toxfinder_study(
    initial, records, prior=elicitation, seed=2028, draws=8, warmup=0
)
assert report.recommendation.observed_patients == 1 and report.recommendation.pending_patients == 1
assert "physical dose pair" in report.to_html()
csv = m.write_toxfinder_subject_csv(records, reference_doses=initial.reference_doses)
assert len(m.read_toxfinder_subject_csv(csv, reference_doses=initial.reference_doses)) == 2
print("Patient CSV/recommendation/report passed", flush=True)
