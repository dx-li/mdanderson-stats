import json

import pytest

from mdanderson_stats.pop_design import PoPDesign
from mdanderson_stats.pop_scenario_file import PoPInputScenario, PoPScenarioInput
from mdanderson_stats.pop_selection_plot import plot_pop_selection


def _input() -> PoPScenarioInput:
    return PoPScenarioInput(
        PoPDesign(target=0.25),
        (
            PoPInputScenario("target at dose two", (0.10, 0.25, 0.45)),
            PoPInputScenario("above target", (0.12, 0.30, 0.50)),
        ),
        total_patients=12,
        cohort_size=3,
        trials=20,
        start_dose=1,
        titration=False,
        earlyterm=True,
        risk_cutoff=0.8,
        seed=175,
    )


def test_scenario_json_roundtrip_runs_existing_report_and_replays(tmp_path):
    original = _input()
    path = tmp_path / "scenarios.pop.json"
    original.write_json(path)
    restored = PoPScenarioInput.read_json(path)
    assert restored.to_json() == original.to_json()

    first = original.run()
    second = restored.run()
    assert first.boundaries == second.boundaries
    assert first.scenarios == second.scenarios
    assert first.seed == 175
    report_path = tmp_path / "report.html"
    first.write_html(report_path)
    assert "target at dose two" in report_path.read_text(encoding="utf-8")


def test_scenario_json_rejects_unknown_or_oversized_inputs_before_run():
    payload = json.loads(_input().to_json())
    payload["simulation"]["surprise"] = True
    with pytest.raises(ValueError, match="exactly the documented"):
        PoPScenarioInput.from_json(json.dumps(payload))
    with pytest.raises(ValueError, match="1 MB"):
        PoPScenarioInput.from_json(" " * 1_000_001)
    with pytest.raises(ValueError, match="nondecreasing"):
        PoPInputScenario("bad", (0.3, 0.2))


def test_selector_plot_keeps_original_dose_positions_and_selected_marker():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selection = PoPDesign(target=0.25).select_mtd([8, 8, 0, 8, 8], [0, 1, 0, 3, 6])
    _, ax = plt.subplots()
    plot_pop_selection(selection, target=0.25, ax=ax, dose_labels=("A", "B", "C", "D", "E"))
    assert tuple(ax.get_xticklabels()[i].get_text() for i in range(5)) == ("A", "B", "C", "D", "E")
    if selection.dose is not None:
        selected = ax.collections[0].get_offsets()
        assert selected[0, 0] == selection.dose
        assert ax.collections[0].get_label() == "Selected MTD"
    assert "Target" in [line.get_label() for line in ax.lines]
    plt.close(ax.figure)
