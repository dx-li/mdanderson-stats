# BARD BF-BOIN post-escalation expansion

`simulate_bf_boin(..., expand_after_escalation=True)` adds the documented
optional expansion path to the BF-BOIN calendar simulation. The option is off
by default. At the end of escalation, it holds the last dose actually treated
for escalation fixed as `c` and continues enrollment at `c - 1`. It stops when
that dose's assigned count, including pending patients, reaches `n_cap`, or
when the dose closes for toxicity. The dose is never redirected to another
level during expansion.

```python
from mdanderson_stats import BFBOINDesign, simulate_bf_boin

result = simulate_bf_boin(
    BFBOINDesign(n_cap=3),
    true_toxicity=[0.0, 0.0, 0.0],
    true_response=[1.0, 1.0, 1.0],
    cohorts=2,
    cohort_size=1,
    trials=1,
    start_dose=1,
    expand_after_escalation=True,
    accrual_rate=0.001,
    rng=21,
)
assert result.assigned.tolist() == [[3, 1, 0]]
assert result.expansion_patients.tolist() == [2]
assert result.expansion_stop_reason == ("assigned_cap",)
```

Expansion uses the existing BF-BOIN activity rule: a dose is eligible after a
response is observed at that dose or a lower dose. If the target dose is not
eligible while relevant outcomes are pending, the calendar processes those
observations before testing again. If the outcomes resolve without activity,
the result reports `activity_unavailable`. A safety stop ends the trial and
does not enter expansion; a precision stop and exhaustion of the configured
escalation cohorts do permit it. If the final escalation dose is dose 1, the
result reports `no_lower_dose`.

The result retains `expansion_patients`, `expansion_stop_reason`, and
`expansion_end` separately from the escalation stop and `escalation_end`.
`trial_duration`, final toxicity counts, final dose selection and patient
histories include expansion and complete follow-up. `expansion_end` is NaN
when expansion was not requested. For a requested expansion that cannot
start, it is the time the explicit reason is established.

The cached BARD guide and its `Expansion` help sheet define the fixed target
dose and the cap/toxicity stopping conditions. The paper's ordinary BF-BOIN
backfill criteria supply the response-activity eligibility rule. They do not
define an asynchronous expansion calendar. This implementation reuses the
simulator's renewal arrivals and processes assessments before same-time
arrivals; that timing is a Python policy. No hidden application default,
BF-BLRM expansion behavior, or stage-two calendar behavior is inferred.

In the current simulator, response assessment uses the same full window as a
negative DLT assessment. Since escalation waits for the current cohort's DLT
assessments, activity at `c - 1` will usually already be known when expansion
begins. The calendar still handles any pending assessments chronologically;
it does not rewind the next arrival if an observation changes eligibility.

The option does not alter default no-expansion simulations. Focused tests
check fixed-dose targeting against the actual last escalation cohort, cap
accounting, no-activity termination and the no-lower-dose case. Source details
and limitations are recorded in the
[expansion audit](../research/bard-expansion-audit.md).
