# ARAND 5.2.2 original-instruction research references

This bundle generates 98 independent reference fixtures from the original ARand.Calcs
library. It contains no vendor executable or copied binary. It does not launch the
Windows application or implement a replacement CLR. It executes bounded portions of
original x86 or decoded original IL with the service substitutions below.

Obtain the official installer lawfully:
https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/ARAND/AdaptiveRandomization_V5.2.2.zip
Archive SHA256: ff9c4d490a8876e1b3197d6fcb809bd8d001761be13f083a667e0e7db34fc877
Extract ARand.Calcs locally. Required library SHA256:
a11d60157236bdf2e3395de39909dbb029b22bae52bcee5dd6d8ee4eebb3a6e3
The MSI extraction originally named that file _8BA1657C80DC5CB78F29D8E43CFB9F20.

Use Python 3.10+ with dnfile, dncil, pefile, and unicorn installed in an isolated
virtual environment. No network is used by the runner.

```sh
python run_reference.py --library /path/to/ARand.Calcs.dll --output-dir ./fixtures
```

Each subprocess has a 120-second timeout. Wrong-hash libraries are rejected before
any original instructions are read. Fixtures are JSONL; manifest.json records their
counts, hashes, and the original library hash. The thirteen families are:

| Family | Original RVA | Fixtures | Execution / supplied services |
|---|---|---:|---|
| x86/floor | 12e1a0 | 7 | x86 tuner; vector element accessor feff0, pow 1a5db0 |
| scheduler | b4604 | 8 | IL CheckTime; counts, pending flag, trial status writes and look callbacks |
| outcome | 13bcc0 | 15 | x86 visible outcome; patient outcome-map lookup13cb40, string comparison102f10, security cookie1a57f8 |
| pending | 13f030 / 13e190 | 6 | x86 patient/endpoint traversal; visible outcome13bcc0, sole endpoint string assignmentfd0e0, security cookie1a57f8 |
| status | 715e0 | 7 | IL CheckMinPrBest; map/string primitives, supplied posterior/futility scalar queries, status getters/writes |
| mincriteria | b4af8 | 6 | IL CheckMinCriteria; containers and trial/arm/group counts/limits |
| minmax | 6f7c4 | 11 | IL CheckMinMax; containers, counts, limits, status changes, posterior/final callbacks |
| final | 70890 | 8 | IL whole-trial final selection; containers, statuses, supplied probabilities, winner writes |
| groupfinal | 6ffa0 | 5 | IL group final selection; same service classes as final |
| allocation | 6d5c8 | 4 | IL weighted endpoint aggregation; maps, supplied probabilities, then original x86 tuner |
| predictive | 70e8c | 5 | IL dormant predictive stopping policy; supplied predictive probabilities, containers/status operations |
| tte_stats | 1b62d0 | 10 | x86 sufficient-stat helper calling original visible-outcome13bcc0 and followup13be40; same three primitive hooks as outcome |
| integrated | b4348 | 6 | IL GetTreatment recursively calls original-instruction scheduler, minima, minmax, status and final interpreters; fixed supplied posterior services |

IL interpreters have 20,000-instruction budgets (scheduler:10,000).
The x86 tuner has 1,000,000 instructions; outcome/pending/tte_stats have100,000.
Budget exhaustion fails rather than producing a fixture. Unsupported instructions or
service calls fail explicitly. Fixture domains are bounded valid inputs, not
exception-handling coverage. The source files expose every service boundary in
`call`/`calli` handlers and x86 hooks. IL locals, map nodes, strings and pointer-like
addresses are interpreter fixtures, not a complete native runtime.

Integration uses explicit `event_log` and `share_statuses` keyword parameters to
connect primitive services, with no dynamic source rewriting or whitespace-based
patching. Posterior calls supply fixed probabilities; no claim is made that these
six fixtures independently validate posterior mathematics, native RNG, trial
enrollment, the entire simulation loop, or formatted desktop output. Allocation
uses supplied eligible-arm selection and returns an allocation-service marker;
original weighted aggregation and tuning execute, but native random selection does
not. See the recovery report for the exact scope distinctions.

The predictive-policy fixtures do not establish predictive inference support:
the standard calculator's predictive vtable implementation is a no-op. Latent
multi-endpoint/container templates do not establish an exposed desktop workflow.
The shipped desktop exposes a single binary OR TTE outcome and no covariates;
covariate UI items are permanently hidden in InitializeComponent.

Verification performed: all98 parsed JSON objects matched the13 earlier scratch
fixture families, including the pending exponential-monitoring path that suspends
all arms then stops through original minmax and final control flow.

TTE sufficient-stat fixtures independently distinguish visible censor0 from event1.
Before record time both yield event0 and accumulating exposure; at/after record
time exposure is capped, but only event1 increments the failure count.

## Python compatibility differential checks

After installing the Python package, the deterministic bounded comparison can be
run alongside the original-reference dependencies:

```sh
ARAND_REFERENCE_LIBRARY=/path/to/ARand.Calcs.dll python verify_python.py
```

It performs 300 status transitions and 600 group/whole selection comparisons
using seed 123, including threshold equalities, permanent closures and conflicting
posterior values. These deliberately include non-normalized supplied best-arm
scores to test controller branches; they do not validate a posterior estimator.
The fixture runner and this check are research tools, excluded from the wheel.

`arand_report_index_reference.py` is a separate 12-case supplement. It executes
only the report's original duration-percentile branch/index/read instructions
(RVA0x60a14, IL927..1092) with supplied sorted vector memory and Math.Ceiling.
The original 98-fixture manifest is unchanged. Invoke with the same library env
variable and compare JSONL to `report-index-supplement.jsonl`. This does not run
the complete report, enrollment sorting or GUI rendering.

`arand_action_reference.py` separately executes original SetAction IL
(RVA0x6f5a4) with supplied string comparison and a status cell. Its eight cases
are preserved in `action-supplement.jsonl`. Original constructor inspection
establishes initial min/max action cells of2; empty/unrecognized action strings
leave the supplied cell untouched. The probe does not execute XML parsing or
claim a complete saved-study loader.

`verify_constraints.py` adds 300 deterministic comparisons (seed314) against the
original ordered arm/group constraint instructions, using the same library and
installed Python package. Cases include conflicting limits, closed statuses,
start/duration equality and reopening; trial limits are not reached. Together
with `verify_python.py` there are1,200 differential comparisons.

`arand_assignment_reference.py` executes GetTreat's original inverse-CDF loop
(RVA0x6ccb4), substituting only the uniform and bounded vector services. Seven
cases in `assignment-supplement.jsonl` cover exact/adjacent cumulative ties and
zero-leading weights. The caller still determines whether the vector is the
eligible population or the all-arm forced population. `supplement-manifest.json`
records hashes and scope for all27 supplementary probes, separate from the
original98-fixture manifest.
