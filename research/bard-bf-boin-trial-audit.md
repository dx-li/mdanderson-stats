# BARD BF-BOIN complete-trial workflow audit

The cached BARD guide describes BF-BOIN dose escalation and backfill, then
stage-two enrollment of eligible stage-one patients followed by covariate
adaptive randomization and OBD selection. The paper's Methods section describes
stage-two minimization and final outcome-based selection; the two-stage target
includes eligible carryover. Existing components already supplied the BF-BOIN
calendar simulator, categorical response model, minimization score, and both
OBD selection methods. The new `bard_bf_boin_trial.py` composes those APIs into
one patient-level trial ledger rather than creating another stage-one simulator
or an artificial adapter for the separate BF-BLRM replay.

The implementation records actual stage-one settings and output, the
calibrated response model and dose truths, the stage-two policy, patient-level
factor/profile rows, outcomes and calendar times, inclusive carryover/count
tables, both final analyses, selection statuses, and a replay seed when one is
available. If no RNG seed is supplied, the generated entropy is captured. When
the caller passes an existing `Generator`, its originating seed cannot be
recovered; allocation sub-seeds are still recorded per new stage-two patient.

## Method boundary

The stage-one calculations and safety history come from `simulate_bf_boin`.
The stage-two assignment and OBD calculations call `bard_minimization` and
`bard_select_obd`. The source defines an inclusive total target and requires
eligible stage-one patients on the two selected doses to contribute to both
imbalance history and joint outcome counts. All such patients are retained if
their number exceeds the target; the Python result marks the overshoot and does
not remove patients to meet a smaller target. The stage-two pair defaults to
the stage-one selected MTD and adjacent lower dose. Callers may specify another
ordered pair after a safe non-null stage-one MTD exists; a pair containing a
persistently eliminated dose is rejected from stage two.

The source identifies two factors for balancing in the published three-factor
example. The Python API exposes the zero-based `balanced_factors` columns and
defaults to all modeled factors for general inputs. It does not introduce
arm-specific quotas. Both utility and noninferiority analyses use the same
combined stage-two pair counts; noninferiority is left unavailable if either
arm has no observed patients.

The stage-two calendar is explicitly a Python policy because the recovered
source does not define stage-two assessment timing. The implementation starts
after stage-one complete follow-up, enrolls the first stage-two patient at that
time, and applies configured renewal gaps to later patients. It uses BF-BOIN's
existing Weibull DLT-window calibration and observes response at the window
end. An optional joint DLT/response probability is validated against the
profile-specific marginals; absent that input, conditional independence is the
documented Python assumption. Stage-two profile weights are conditioned on
the stage-two eligible profile set, while response intercepts retain their
original calibration distribution.

Native pair-selection controls, hidden design defaults, exact RNG identity,
native calendar rules, and saved protocol/report formatting remain outside
this implementation. The community workflow is statistical orchestration and
an auditable in-memory ledger; it does not claim native report parity. See
[`docs/bard-bf-boin-trial.md`](../docs/bard-bf-boin-trial.md) for use and the
existing [BARD source crosswalk](../docs/bard-sources.json) for cached source
records.
