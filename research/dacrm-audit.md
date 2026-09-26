# DA-CRM source and numerical audit (2026-09-26)

Active next method after BMA-CRM complete-outcome posterior/decisions. The
main baseline is 263c420. The full software goal remains open. Root branch is
feat/dacrm; sole Luna child stplan_core uses mda-efftox-core on feat/dacrm-core.

Sources:
- https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/CRMSuite/DA-CRM_Description.pdf
  Liu,Yin,Yuan2013, Section2.3: joint binary/time likelihood, Bernoulli missing
  outcome imputation and gamma hazard full conditionals. Section2.4 restricts
  DA moves to one level. Paper examples use nine hazard intervals.
- https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/CRMSuite/BMA-CRMSimulatorHelp.pdf
  AppendixI: six hazard intervals from trimester probabilities; endpoint CDF.99.
  AppendixII: extra trial-conduct rules. Native gamma dispersion is not separately
  specified in this guide, so trimester calibration must take it explicitly.

Posterior inputs: one prior-median skeleton, zero-based patient doses, outcomes
-1pending/0completed-noDLT/1observedDLT, and patient follow-up or DLT times.
Prior supplies interval endpoints including0/window, positive gamma shape/rate,
and alpha normal SD. Pending times are before window; nonDLT times equal window.
DLT times are in[0,window], including a reported event on the treatment date
as permitted by the CRM Suite guide. An event at an interior cut belongs to the
interval on its right, while the final interval includes window.

Exposure s[i,k]=max(0,min(time[i],break[k+1])-break[k]). Conditional pending
event odds equal pi/(1-pi) multiplied by exp(-sum(lambda*s)). For imputed y,
alpha updates under the complete Bernoulli likelihood, and lambda[k] updates
with shape+observed DLT count and rate+sum(y*s). A pending imputed toxicity
contributes survival exposure through current follow-up, not an event count.
Completed noDLT patients contribute no hazard exposure. Use the published
working likelihood; adding a time-truncation normalizer would change it.

Core implementation delegated to Luna: dacrm.py/test_dacrm.py. Explicit NumPy
Generator, serial chains, scalar elliptical slice for alpha and gamma Gibbs
updates. Reuse stable BMA log terms and existing ChainSummary diagnostics.
Retain immutable draws and pending conditional probability draws. Direct prior
sampling is exact when all patients are pending with zero follow-up, or no
patients exist. No desktop random-seed/hidden-MCMC-setting parity claim.

Exact independent reference: tools/reference_dacrm.R enumerates at most
three pending binary outcomes for synthetic scenarios; integrates gamma hazards
analytically and alpha with scalar base-R quadrature. It uses patient Bernoulli
likelihoods, avoiding incorrect grouped-binomial reweighting of completions.
Six cases: complete data, early/late pending, all pending at zero follow-up,
no patients, and DLTs at interval boundaries. Compare posterior means, overdose
probabilities and pending probabilities using chain diagnostics and Monte Carlo
standard errors, not arbitrary exact-equality assertions for MCMC.
Reference generation passed with warnings treated as errors (0.549s,86.09MiB,
zero swaps). Prior identities, conditional gamma moments, and deterministic
complete-data CRM agreement were checked separately. Root integration checks
passed with warnings as errors: ten test functions covering six independent
reference cases, prior calibration, bounded work, and numerical extremes. The
run took 8.46 seconds with peak process memory 132.59 MiB and zero reported
process swaps. Earlier pre-scaling run peaked at 134.27 MiB. All numerical jobs
and child work remained serial. No full suite or large simulation was run.

Integrated convenience helpers (implemented by Luna): paper uniform hazard
prior with default nine intervals and dispersion2; current six-interval
trimester calibration with required dispersion. Prior means /dispersion gives
gamma shapes; reciprocal dispersion gives rates. Under a time-unit change,
dispersion must scale as inverse time. Reject q5>=.99 in trimester calibration.

Numerical review identified finite large Gamma draws whose unscaled sample
variance overflowed. Local parameter scaling now avoids this overflow; tiny
nonzero draws are scaled too. Undefined R-hat for constant underflowed draws is
retained as a diagnostic, while nonrepresentable moments fail explicitly.
Gamma scale and draws are checked, posterior rate sums cannot silently overflow,
and the public conditional-probability helper checks its broadcast size.

Decision extension was implemented by Luna and reviewed at root. The paper profile uses one-level
interim movement, default safety cutoff 0.96 and unrestricted final choice.
The CRM Suite profile uses the guide's cutoff 0.9 and requires an explicit
minimum-observed count. While any outcome is pending it adds an observation
gate for escalation, plus the raw observed
rate restriction and final three-treated-patient fallback without skipping
untried doses. Observation gating precedes the raw-rate cap: this precedence is
a disclosed Python convention where the guide leaves ordering unresolved.
Table 3's published esophageal trial posterior means provide a deterministic
allocation-sequence check; these are not synthetic native posterior outputs.

Remaining beyond this batch: complete-data deterministic CRM routing, pending-
outcome look-ahead, actual/calendar conduct and operating characteristics,
native files/reports. Online entry 133 remains pending. Catalog entries 81 and
132 stay partial; overall counts remain 62 implemented, 56 partial, 20 pending.
GitHub publishing remains blocked by the session's approval policy. Do not
retry a write using an alternate transport to bypass that restriction.


Integration validation completed:
- Root decision checks: seven tests plus complete-data gate bypass and unrestricted
  paper final-choice examples; 0.11 seconds for tests, peak 136.05 MiB, zero
  reported process swaps. Together with the posterior/prior/reference suite,
  seventeen focused test functions pass.
- Ruff check and format pass for all eight affected Python files; targeted mypy
  passes for the three new source modules. Diff whitespace checks pass.
- Cached Hatchling built wheel and source distribution without installing
  dependencies. Isolated wheel import verifies all eight public exports,
  source/catalog byte equality, partial entry statuses, and exclusion of raw
  original downloads. Every Python example in docs/dacrm.md executes from the
  wheel; peak 116.19 MiB and zero reported process swaps.
- Full CI was not run; validation stayed focused on statistical correctness,
  scientific policy behavior and package integration, per user preference and
  OOM precautions. No CI workflow was changed.
- Highest measured process memory in this batch was 136.05 MiB. Latest system
  memory-pressure report showed 54 percent available. These are measurements,
  not a guarantee about other applications' memory use.
