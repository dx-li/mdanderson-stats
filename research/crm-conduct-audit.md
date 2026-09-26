# CRM Suite conduct audit

Baseline: main 3e338f6. The preceding goal turn made verified progress by adding
DA-CRM posterior inference, prior calibration and dose decisions. The full
catalog goal remains active. Root uses feat/crm-conduct; the sole Luna child
uses mda-efftox-core on feat/crm-conduct-core. Numerical jobs remain serial with
one BLAS thread; no full suite, installations or large simulations are planned.

Primary source: CRM Suite guide version 1.0.0 (2018), Appendix II and trial
conduct sections:
https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/CRMSuite/BMA-CRMSimulatorHelp.pdf

Look-ahead applies to CRM/BMA-CRM, not DA-CRM. It permits action only when all
pending-outcome completions yield the same action and dose. Grouping missing
binary outcomes by dose reduces the enumeration from patient identities to
count combinations, with no loss for this count-based model. Endpoints may
prove disagreement, but agreement still requires all mixed completions.
Work-limit results must be explicit waits, not declarations of invariance.
Refits use original priors and complete counts; posterior model weights cannot
serve as new prior weights because that would count observed data twice.

Root found that normalized original priors can themselves underflow to zero
for extreme finite relative weights. fit_bmacrm already preserves these weights
internally in logarithmic calculations, but its output needs the original raw
weights as well for lossless later refits. The existing independent R
rescued_prior case supplies a useful high-information benchmark for this fix.

Calendar scope for this batch is stateless reconstruction from fully ascertained
historical or simulated outcomes, plus an inference/decision router. Positive
infinity means a confirmed no-DLT full window, never unknown or missing
ascertainment. Future enrolled patients and not-yet-observed DLTs must not leak
into earlier snapshots. Boundaries include events at treatment and window end.
No assumptions about database ingestion/correction times are made.

The DA branch samples only with pending outcomes. Complete outcomes use
single-skeleton deterministic integration with the DA prior's alpha SD and the
ordinary CRM conduct policy. The route is reported explicitly and complete
cases must consume no RNG draws. Snapshot and method consistency are checked
before numerical work. Future full trial loops must discard retained MCMC
draws between steps rather than accumulating an unbounded posterior history.

Remaining after this scope: cohort/event scheduling, trial simulation and
operating characteristics, native file/report workflows, and independent online
entry 133 conventions. Publishing is still blocked by the session approval
policy; do not bypass that denial with another transport.


First checkpoint integrated as 03d9635: look-ahead plus immutable original
model-prior retention. Root checks passed with warnings as errors: eleven test
functions including the independent base-R posterior fixture; 3.60 seconds,
136.41 MiB peak process memory, zero reported process swaps. The extreme prior
case favors the rescued model and treats at the lower dose; losing its original
weight would incorrectly trigger a safety stop. Empty-trial direct delegation
and the documentation's invariant-look-ahead example were checked separately.

Second checkpoint integrated as df52e3e: immutable calendar snapshots and
complete-data/look-ahead/DA routing. Root caught an endpoint rounding defect
during review: with enrollment 0.4 and a delay/window of 0.1, subtraction can
produce follow-up below 0.1 at calendar time 0.5. Classification now compares
representable calendar endpoints, with a focused event/non-event regression.
Six calendar test functions passed with warnings as errors in 1.45 seconds;
132.58 MiB peak process memory and zero reported process swaps. Ruff checks
and formatting passed for the six affected Python files; targeted mypy passed
for the three affected implementation modules.

Built both distribution formats with the cached Hatchling runtime. An isolated
wheel import verified six new public exports, source/catalog byte equality,
and exclusion of raw downloaded material from the distributions. All three
Python examples in docs/crm-conduct.md ran successfully. The same isolated
check confirmed the empty DA-request start uses deterministic CRM and a snapshot
with inconsistent counts is rejected. It took 2.897 seconds, with 116.39 MiB
peak process memory and zero reported process swaps. No full-suite run, new CI
job, dependency installation, or large simulation was performed.

The online BMACRM surface was audited enough to identify additional BMS and
skeleton-calibration work; its conventions remain unverified and entry 133
remains pending. The older desktop guide's DA safety wait differs from the
newer CRM Suite written policy; this distinction is now explicit in the user
documentation. Catalog counts remain 62 implemented, 56 partial and 20 pending.
This batch advances two partial CRM entries without claiming full coverage.
