"""Run and reopen both current mTPI tasks without original proprietary assets."""

from mdanderson_stats import (
    MTPIDesktopStudy,
    MTPIDesktopTableStudy,
    mtpi_desktop_simulation_report,
    mtpi_desktop_table_report,
    read_mtpi_desktop_report,
    simulate_mtpi_desktop,
)

study = MTPIDesktopStudy(0.3, 0.05, 0.05, 12, 3, 1, ((0.1, 0.3, 0.6), (0.25, 0.5, 0.7)), 100)
for profile in ("safe", "desktop_legacy"):
    result = simulate_mtpi_desktop(study, profile=profile, seed=72)
    saved = read_mtpi_desktop_report(mtpi_desktop_simulation_report(result))
    assert saved.study == study and saved.profile == profile and saved.seed == 72
    print(profile, result.selection_probability.tolist())
monitoring = MTPIDesktopTableStudy(maximum_patients=12)
assert read_mtpi_desktop_report(mtpi_desktop_table_report(monitoring)).study == monitoring
