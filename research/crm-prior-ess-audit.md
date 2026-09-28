# CRM prior effective sample size

This is the complete-outcome empiric CRM branch of BayesESS entry 154.
Sources inspected September 28, 2026:

- BayesESS 0.1.19, `essCRM` in
  [internal.R](https://github.com/github-js/BayesESS/blob/4bbf4df3789912b967774e8ff5c3a2d6d5646cdd/R/internal.R).
- CRAN dfcrm 0.2-2.1, `crm`, `onetrial` and `crmsim` in
  [dfcrm.R](https://github.com/cran/dfcrm/blob/18891ccb969e3e4f87e4489a04b48df227bb9273/R/dfcrm.R).

The pinned dfcrm file is 68,519 bytes, SHA-256
`cef859c8d5ba5a247679e2664d38f94c6c1314e0a67facfa155f0037b3a90341`.
Native source is retained as an ignored research input, not redistributed.

## Native model and adaptive experiment

The model is `p_j(beta) = skeleton_j ** exp(beta)`, with
`beta ~ Normal(0, beta_sd**2)`. The supplied `PI` vector is the true toxicity
probability, distinct from the working skeleton. Trials start at dose 1 with
single-patient cohorts and run for the specified full size M.

Each interim estimate is the posterior mean of beta. Plugging that mean into
the dose-response curve determines the dose closest to target, with the first
index breaking a tie. This is different from integrating the dose-response
probability over the posterior. After a DLT, the next dose cannot exceed the
last dose; otherwise it cannot exceed the last dose plus one. Downward moves
are unrestricted. The final closest-to-target recommendation is unrestricted.

For each m from 0 to M, native BayesESS reruns the same full-M trials and
samples m patients without replacement from each complete trial. It evaluates
their log-likelihood second derivatives at beta=0. These are subsets of the
full adaptive experiment, not prefixes or new trials with m patients.

For realized per-patient information I_i, the exact conditional expectation
over subsets is `(m/M) * sum(I_i)`. Averaging this quantity across simulated
trials removes the redundant subset-sampling noise and repeated refits while
preserving the target expectation. A resulting linear information path is an
expected subset path; it is not the information accumulated along the adaptive
patient sequence. Across-trial simulation variability still remains.

The source information gap is `1/beta_sd**2 - expected_information(m)`.
Unlike the regression calculator, this CRM implementation does not subtract
the epsilon-prior information. Native `approx` evaluates the gap on 50 points
spanning [0,M] and chooses the first nearest-to-zero value. That grid value is
not an exact continuous crossing and can equal M even if no crossing exists.

## Native posterior moment discrepancy

`dfcrm::crm` integrates the posterior denominator over the full real line,
but integrates the mean and second-moment numerators only over [-10,10].
For the default `beta_sd=sqrt(1.34)`, the lost Gaussian tails are negligible.
For diffuse priors they are material and do not define a consistently
normalized truncated posterior either.

The fixed-data reference uses skeleton [.05,.15,.30,.50], target .25 and
levels [1,2,2]. With beta_sd=4 and outcomes [0,0,0], the native mean is
2.7957516434 while consistent full-real integration gives 2.9267207125.
With outcomes [1,1,1], these are -4.3832498965 and -4.6361524300.
Any implementation using consistent full moments must disclose this correction;
unqualified numerical parity across diffuse priors is not warranted.

## Independent small reference runs

`tools/reference_crm_prior_ess.R` sources both pinned files without installing
packages. It extracts and evaluates the original nested `getDiffCRM`
definition. Only the Bernoulli outcome generator in dfcrm's function environment
is replaced by supplied uniforms; inference, allocation and final selection
are unchanged.

Three scenarios (mixed toxicity, all-safe and all-toxic) each run three
12-patient trials. The fixtures retain 108 uniform/outcome/dose/prepatient-beta
rows, nine trial endpoints and 39 expected-subset information rows. Endpoint
information uses the original BayesESS derivative. The ordinary scenario's
continuous ESS is 1.3059566036; the native 50-point matching convention yields
1.2244897959 for the same expected information path.

Six fixed-data references separately record native moments and full-real
moments at default and diffuse prior scales. Full-real reference integration
uses relative tolerance 1e-10. Native reference calls retain the original
default integration tolerances. The complete R run took about .4 seconds;
no installation or large simulation was required. Python comparisons remain
pending.

## TITE branch remains separate

The pinned simulator confirms that `arrival` contains enrollment times. The
BayesESS TITE branch nevertheless uses `arrival/obswin`, capped at one, as a
follow-up weight, and hardcodes Poisson accrual despite its `accrual` argument.
The actual dfcrm interim likelihood uses elapsed follow-up and observed DLTs.
Weighted non-DLT log likelihood can also have positive second derivative;
nonpositive curvature cannot be assumed for every TITE observation.
These issues require an explicit assessment-time/weight convention before the
TITE workflow can be described as a corrected implementation.
