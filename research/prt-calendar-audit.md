# PRT calendar workflow source audit

Entry 69 already implements the interval likelihood, state-space posterior,
covariance-weighted isotonic transformation, predictive probabilities, conduct
rules and final selection. This audit identifies the remaining calendar scope;
it is not a completion claim.

The primary method is Bekele et al., [Biostatistics 9 (2008), 442–457](https://doi.org/10.1093/biostatistics/kxm044).
Local source text is retained under ignored `research/raw/PRT/`: `paper.txt`
section 4.3, `conduct.txt` lines 16–27 and 60–76, and `sims.txt` lines 36–93.

Conduct rules run after cohort enrollment. Suspended accrual is reconsidered
on arrivals and observed follow-up changes. The existing decision kernel
handles stopping, de-escalation, staying and escalation without skipping an
untried dose. The final selector does not exclude untried doses. Calendar replay
must convert only observed events and completed intervals into the likelihood;
an incomplete interval contributes neither count.

Cohort size, initial dose, arrival process and waiting/abandonment policy need
explicit inputs. The guides specify average arrival time but no verified
arrival distribution or FIFO rule. Final analysis after all enrolled patients
complete follow-up is a declared replay convention where exact native timing
is unverified.

The paper delegates its piecewise-exponential toxicity generator to Supplement
Appendix B. The simulation guide names early/late parameters alpha and beta
without their equations. A bounded primary-source lookup found listed
supplements `kxm044_1.pdf` and `kxm044v2_1.pdf`, but their contents were
inaccessible. Cutpoints, rates and parameterization remain unverified and must
not be invented to claim native operating-characteristic parity.

The [existing projection limitation](../docs/prt.md) is also material: full
inverse-covariance weighting can produce probabilities outside [0,1] for the
guide-history pilot. A replay must propagate that failure clearly. Resolving
native safeguards is necessary for a complete end-to-end PRT claim.
