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

Multiple-endpoint calendar simulation, calibration and app parity require their
own implementation evidence; these references do not establish them. The binary
calendar API retains its existing uniform analysis-weight convention.
