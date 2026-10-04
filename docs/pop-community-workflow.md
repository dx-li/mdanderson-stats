# PoP selection plot and portable simulation inputs

PoP's executable `select.mtd.pop` returns an MTD and isotonic dose-toxicity
estimates, and `plot.pop` plots those estimates against the target. The
optional Python plot shows eligible estimates at their original dose numbers,
highlights the selected dose, and draws the target. Untreated and
safety-excluded doses remain absent rather than shifting later dose labels.
Matplotlib is imported only when plotting is requested.

```python
from mdanderson_stats import PoPDesign, plot_pop_selection

design = PoPDesign(target=0.25)
selection = design.select_mtd(patients=[8, 8, 0, 8, 8], toxicities=[0, 1, 0, 3, 6])
axes = plot_pop_selection(
    selection,
    target=design.target,
    dose_labels=("1", "2", "3", "4", "5"),
)
axes.figure.savefig("pop-mtd.png", dpi=120)
```

The native help mentions 95% credible intervals, but the pinned executable
selector and plot do not compute or display intervals. This plot adds none.

## Save, reopen, and run scenario inputs

`PoPScenarioInput` stores the actual design, monotone toxicity scenarios,
simulation settings, and root seed in a versioned JSON document. It can be
reopened and passed to the existing bounded report simulation; the file is
data only and cannot execute code.

```python
from mdanderson_stats import PoPDesign, PoPInputScenario, PoPScenarioInput

request = PoPScenarioInput(
    design=PoPDesign(target=0.25),
    scenarios=(
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
request.write_json("pop-scenarios.json")
report = PoPScenarioInput.read_json("pop-scenarios.json").run()
report.write_html("pop-report.html")
```

The report retains final-selection probabilities in no-MTD-then-dose order.
Use `plot_pop_selection_percentages` to draw that scenario's selection
distribution. The Python plot converts the stored probability fractions to
percentages, includes no-MTD as its own bar, and accepts optional labels for
the dose bars.

```python
from mdanderson_stats import plot_pop_selection_percentages

selection_ax = plot_pop_selection_percentages(report.scenarios[0])
selection_ax.figure.savefig("pop-selection-percentages.png", dpi=120)
```

The JSON records the cutoff settings and random seed used by the Python
simulation. Scenario vectors preserve original dose order; the selected dose
and plot labels use those same indices. The run still follows the existing
Python simulation and seed convention, not an R random stream or native app
file format. Input size, dose count, scenario count, and aggregate simulations
use explicit limits; a saved input does not bypass them.
