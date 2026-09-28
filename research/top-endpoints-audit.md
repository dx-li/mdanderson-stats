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
and 9.15 before rounding. Python comparisons remain pending.

General calendar simulation, calibration, final-action policy and app parity
require their own implementation evidence; these references do not establish them.
