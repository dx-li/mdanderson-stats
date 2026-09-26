# CRM model-selection coverage audit

Baseline: main 7652e45. The previous goal turn made progress: CRM trial replay
and serial operating-characteristic simulation were integrated, with focused
numerical and isolated-wheel validation. The global catalog/publishing goal
remains incomplete and active.

One Luna implementation agent works in the existing mda-efftox-core checkout.
Root handles source verification, independent reference comparisons, docs,
catalog reconciliation and integration. Only one numerical job runs at a time,
with OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1. No dependency installation,
full-suite run or large simulation study is needed for this batch.

Primary sources verified on 2026-09-26:

- https://saasresearch.hku.hk/~gyin/materials/2009YinYuanJASA.pdf
  Section 2.3 defines BMS and strict Occam-window membership, recomputed as data
  accrue. Section 3 uses prior alpha SD 2. The existing Suite decision rules
  differ from the paper's one-level escalation/deescalation algorithm.
- https://biostatistics.mdanderson.org/shinyapps/BMACRM/
  Version 1.0.2.0, updated 2025-12-15; visible BMA/BMS options and complete-count
  trial-conduct inputs. It references Yin/Yuan and Pan/Yuan. No native output
  comparison or hidden-prior verification was performed.

Contract: keep actual posterior model probabilities separate from aggregation
weights, select/trim in log space, and use the chosen mixture for both dose
means and safety probabilities. The first exact BMS tie wins. Occam requires
an explicit threshold in [0,1), uses a strict inequality and renormalizes.
Retain original raw priors for every future fit and hypothetical completion;
temporary exclusion must not permanently remove a model. These parameters
are rejected for DA-CRM rather than silently ignored. Existing BMA defaults
and numerical budgets remain unchanged.

The independent aggregation comparison uses the already generated base-R
posterior fixture for mixed, zero-prior and rescued-prior scenarios. It selects
or renormalizes those reference model probabilities and combines independently
integrated model means and overdose probabilities. This is not native app
parity or a reproduction of large published simulation tables.

Calibration follow-up: the indexed Pan/Yuan article at
https://pmc.ncbi.nlm.nih.gov/articles/PMC5026535/ exposes the Lee/Cheung recursion
and the ranking-then-simulation procedure, but its displayed Q equation has an
inconsistent equality. The author code index
https://odin.mdacc.tmc.edu/~yyuan/index_code.html does not currently expose the
promised calibration function. Verify the regression convention before
implementing Q; a highest-Q set alone is not the full optimal calibration.

## Integrated validation

Luna commits d18a6f5 and 255351a were integrated as 13bbcbf and 50d24c8.
Luna's final 23-test run covered aggregation, the existing BMA posterior and
decision checks, calendar routing, trial replay and serial simulation. Ruff
and targeted mypy passed. The later test-name-only clarification received a
Ruff check without rerunning the numerical checks.

Root ran the two independent BMA reference tests and the complete-trial R
reference test: three passed with warnings as errors in 5.11 seconds, peak
135.31 MiB, zero reported process swaps. All six existing look-ahead tests
also passed with warnings as errors in 2.52 seconds, peak 134.17 MiB and zero
reported process swaps. Root's new reference test passed Ruff check/format.
No full repository suite, large simulation study or new CI job was run.

The model-selection test now uses a multi-model calendar example with different
BMA and BMS final choices. It does not mistake a single-model run for evidence
that aggregation affects decisions. Simulation preflight uses the same strict
threshold parser as the posterior fit, so malformed thresholds fail before
patient randomness is consumed. Existing analytic evidence for a prior-only
fit already ensures exact prior ties, and was preserved.

The wheel and source distribution built with cached Hatchling. Isolated-wheel
checks verified the five changed module bytes and catalog, executed both new
documentation examples, and confirmed malformed threshold and unsupported DA
aggregation fail without consuming the scenario RNG. Raw upstream downloads
remain excluded. This process took 3.37 seconds, peaked at 115.66 MiB and
reported zero process swaps. A separate memory-pressure reading reported 38%
system-wide free memory; process measurements do not describe every permitted
simulation workload.

Catalog counts after this batch: 62 implemented, 57 partial, 19 pending (138
entries). Online entry 133 moves from pending to partial. Native conventions
and automatic skeleton calibration remain open. MTADF entry 114 is the next
uncovered scientific implementation target; see mtadf-next-audit.md.

GitHub publication remains blocked by the earlier approval-required tool error
under the session's never-approve policy. No alternate write transport was
attempted. The broader goal remains active because scientific coverage can
continue locally; this publishing restriction is not a global impasse yet.
