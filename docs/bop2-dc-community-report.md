# BOP2-DC community reports

The report facade converts an already constructed BOP2-DC design and its
existing exact operating-characteristic or bounded simulation results into an
immutable Python report object. It can render a portable static HTML file.
The cached source set contained no BOP2-DC app HTML or native report schema,
so the document records the Python inputs and outputs and does not claim native
format, control, RNG, or optimizer parity.

```python
from mdanderson_stats import bop2_dc_design, bop2_dc_binary_report

design = bop2_dc_design(
    20,
    lrv=0.2,
    cmv=0.3,
    prior=(0.5, 0.5),
    looks=[10, 20],
    lambda_lrv=0.9,
    lambda_cmv=0.5,
    gamma_lrv=0.5,
    gamma_cmv=0.5,
)
report = bop2_dc_binary_report(
    design,
    [
        ("reference", 0.2),
        ("promising", 0.4),
    ],
)
report.write_html("bop2-dc-report.html")
```

The binary and paired exact APIs report exact finite-sample OCs. Normal,
survival, categorical, and randomized simulation reports use the existing
serial Monte Carlo engines and display the supplied scenario seed, actual
design settings, trial denominator, action counts/probabilities/MCSE, and
available enrollment, event, exposure, quadrature, or replay-seed summaries.
An MCSE is shown only where the underlying engine computes one. Per-look
simulation action probabilities and MCSEs use all simulated trials; the report
also displays how many trials reached each look. Thus per-look rates are
unconditional probabilities, not conditional-on-reaching rates.

Randomized reports include the fixed assignment tape. Categorical reports
include the category-to-endpoint indicator mapping; randomized paired reports
use the core's joint endpoint-category order. For multiple efficacy, the order
is both, endpoint 1 only, endpoint 2 only, neither. For efficacy/toxicity, it
is efficacy and toxicity, efficacy without toxicity, toxicity without
efficacy, neither. Endpoint 2 is toxicity and uses the core's lower-is-better
convention. Each factory accepts at most 20 scenarios and
bounds aggregate simulated patient paths before invoking the engines. The
existing engines apply their own additional work limits. Split larger batches
into separate calls.

These are Python analysis/report inputs, not a native app input-file contract.
They do not calibrate supplied cutoffs or invent a generic optimizer. To
reproduce a report, keep the design arguments, scenario inputs, and seeds with
the HTML file; Monte Carlo child-seed paths are included where the engine
returns them.
