"""Small executable conduct/simulation example; short chains are demonstration only."""

from dataclasses import replace

from mdanderson_stats import (
    DoseScheduleDesktopSampler,
    DoseScheduleDesktopStudy,
    analyze_dose_schedule_desktop,
    dose_schedule_desktop_analysis_report,
    dose_schedule_desktop_simulation_report,
    read_dose_schedule_desktop_report,
    simulate_dose_schedule_desktop,
)

study = DoseScheduleDesktopStudy(
    (0.15, 0.3),
    (4, 6),
    (10, 12),
    2,
    3,
    14,
    28,
    0.3,
    0.4,
    0.8,
    maximum_patients=3,
    repetitions=2,
    scenarios=(((0.1, 0.2), (0.3, 0.4)),),
)
sampler = DoseScheduleDesktopSampler(draws=16, warmup=8, chains=2)
prior = analyze_dose_schedule_desktop(study, seed=75, sampler=sampler)
prior_report = dose_schedule_desktop_analysis_report(prior)
assert read_dose_schedule_desktop_report(prior_report).study == study
simulation = simulate_dose_schedule_desktop(study, seed=75, sampler=sampler)
report = dose_schedule_desktop_simulation_report(simulation)
assert read_dose_schedule_desktop_report(report).seed == 75
assert replace(study, study_time=1).study_time == 1
print("Prior/conduct report and all-scenario simulation report reopen successfully.")
