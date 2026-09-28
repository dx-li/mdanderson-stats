# TOP multiple endpoints: source contract

The [publisher's Supplementary Methods](https://academic.oup.com/jnci/article/112/1/38/5423189)
(`jnci_112_1_38_s3.pdf`, 26 pages) became readable on September 28, 2026.
Access was through its public article link; PDF screenshots remained unavailable.

Equations 2 and 6 aggregate a joint Dirichlet prior into endpoint-specific
Beta distributions. For endpoint j, a_j=sum(beta_jk*alpha_k),
b_j=sum((1-beta_jk)*alpha_k). Its fractional posterior is
Beta(a_j+observed_events_j, b_j+ESS_j-observed_events_j).
Different ESS values do not define a common fractional Dirichlet posterior.

Supplementary Tables 3/6 specify endpoint-action combination:

| Endpoint states | Co-primary efficacy | Efficacy/toxicity |
|---|---|---|
| One acceptable, one suspended | Continue | Suspend |
| One unacceptable, one suspended | Suspend | Stop |
| Both acceptable | Continue | Continue |
| Both unacceptable | Stop | Stop |
| Both suspended | Suspend | Suspend |

Tables 1/2 use N=45, C=.94, gamma=.50; Tables 4/5 use
N=40, C=.50, gamma=.60. The latter reverse the toxicity posterior tail.
Pages 6–7 additionally specify mixture-uniform timing over window thirds.

`tools/reference_top_endpoints.R` independently evaluates Beta tails and
crossings in base R. It produces 98 posterior rows and 394 complete-count
rows. Selected crossings reproduce the printed values 10.65, 12.32, 15.84
and 9.15 before rounding. The four focused Python tests pass against all 98
posterior and 394 boundary rows. They also cover the timing mixture, patient
summaries, endpoint-specific suspension, batched inputs and retention of tiny
positive prior shapes. Derived marginal shapes that underflow are rejected.

The public guide example was executed successfully. A separate integration
check covered all nine endpoint-status combinations in each mode, at interim
and final looks (36 cases). It checked the source's final combination rule,
including co-primary success with one unresolved endpoint and efficacy/toxicity
termination with the other endpoint unresolved. Two time-unit rescalings
(1e-100 and 1e100) preserved patient-level probabilities and decisions; known
endpoint outcomes correctly ignore irrelevant follow-up values.

The guide and integration check used 116 MiB peak process memory with zero
swaps. Focused Ruff formatting/lint and mypy checks passed. No broad test suite
or CI change was needed for this checkpoint.

The checks above establish the monitoring kernel. Calendar replay, simulation
and binary timing have separate evidence below. Multiple-endpoint calibration
and native app parity remain unfinished.

## Exact references for the calendar extension

`tools/reference_top_multiendpoint_power.R` enumerates four-cell multinomial
counts for eight small, fully observed final-look scenarios. The fixture
`tests/fixtures/top-multiendpoint-power.csv` includes three joint-association
settings per mode with unchanged endpoint margins, plus two null scenarios.
The existing Python decision kernel independently reproduced all four terminal
action probabilities over 1,076 count states; maximum absolute disagreement
with base R was below 1e-13.

These are complete-outcome reference probabilities. A delayed calendar can
terminate while another endpoint remains unresolved under the source's final
combination rule, so its individual stopping-reason frequencies need not match
the fully observed joint classification. The final-only success probability
is suitable for comparison; intermediate calendar paths need separate checks.

## Calendar replay and simulation

`run_top_multiendpoint_trial` and `simulate_top_multiendpoint` now share a
calendar engine vectorized across trials. Each scheduled look uses only outcomes
and follow-up observed at that time. Suspension advances to the next event or
window completion; accrual restarts its next planned gap upon resumption. The
simulator draws joint binary outcomes from the supplied four-cell probabilities,
then independently draws endpoint event times conditional on that joint outcome.
This timing-independence assumption and the calendar conventions are explicit
Python choices, not evidence of native scheduling or random-seed parity.

The implementation's three focused tests passed. Hand-calculated paths cover
co-primary futility, co-primary final success with the other endpoint still
pending, and efficacy/toxicity termination with an unresolved endpoint. Checks
also cover 1e100 time rescaling and preservation of caller-owned timing arrays.
A 20,000-trial final-only simulation agreed with the independent multinomial
success reference within five Monte Carlo standard errors.

Root integration executed both public guide examples, checked read-only truth
timing inputs, repeated-seed results and the partition of terminal actions. The
public examples and checks took 0.020 seconds after package import, with 116.9 MiB
peak process memory and zero process swaps. The aggregate simulator retains
compact trial summaries, not complete histories, and checks its two-million
endpoint-patient-cell limit before allocation.

## Nonuniform binary analysis timing

Binary TOP now accepts the same published three-part conditional timing mixture.
It applies those CDF weights in follow-up evaluation, calendar replay and every
calibration candidate. The generated timing truth remains independently
configured. Equal mixture masses preserve the original uniform-weight path.
Posterior and boundary arithmetic now subtract integer failures before adding
tiny positive prior shapes, avoiding cancellation of the prior.

The worker's 13 focused binary TOP, calendar, calibration and timing checks passed.
The public nonuniform-timing example independently reproduced ESS 9.75 from
seven observed patients and pending weights .75, 1, 1. Root Ruff lint/format and
mypy checks passed for the changed TOP code. A broad suite was not rerun for
these additions; the checks target the changed numerical and calendar behavior.
