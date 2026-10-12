# ARAND 5.2.2 controller recovery — October 12, 2026

This is source evidence for the next implementation batch. It does not promote
Adaptive Randomization from partial or change the existing explicit-policy Python
controller. The recovered contract differs materially from that policy.

## Source and scope

The official `AdaptiveRandomization_V5.2.2.zip` SHA-256 is
`ff9c4d490a8876e1b3197d6fcb809bd8d001761be13f083a667e0e7db34fc877`.
The inspected native/managed library SHA-256 is
`a11d60157236bdf2e3395de39909dbb029b22bae52bcee5dd6d8ee4eebb3a6e3`.
Original binaries remain outside the distribution.

The guide advertises one binary or one time-to-event endpoint. Managed desktop
creation paths `NewBinary_NoCovariates` (RVA `0xcb18`) and
`NewTTE_NoCovariates` (`0xcdb0`) match that scope. `InitializeComponent`
(`0xb7a4`) explicitly hides both covariate creation controls; their fields occur
only in that initializer, with no later method unhiding them. The default trial
constructor creates one outcome. Latent multi-group/endpoint templates are not
an additional advertised desktop workflow. Windows GUI execution was not used.

## Controller findings

- `Director.GetTreatment` (`0xb4348`) checks monitoring when minimum criteria
  permit, then always checks enrollment/duration and arm constraints, then stops
  or allocates. Candidate times drive all monitoring; no independent timer fires.
- Six monitoring modes are present: continuous, integer-time, supplied-time,
  patient-count, exponential batches and observed-outcome count. Exact time/count
  boundaries and threshold advancement differ among modes. Integer equality can
  repeat a look; discrete thresholds skip only strictly smaller values.
- Exponential monitoring requires enrolled count equal to its threshold.
  Pending outcomes call `Trial.SetStatus(1)`, which writes temporary-closed status
  to **every arm/group**. This is not a separate accrual-pause flag. The subsequent
  constraint check can see zero active arms and stop the trial. The nonpending
  branch reopens all arms before analysis. Minimum-enrollment overrides may reopen
  arms again during constraint checking.
- Exponential batch construction starts with `ceil(a)` and appends
  `ceil(a*exp(-b*N))`, updating cumulative enrollment N. Reset instead initializes
  the first threshold with `int(a)`. A noninteger a therefore exposes an original
  inconsistency that must not be silently repaired.
- An outcome is visible at `now >= absolute_outcome_time`; before then it is -1.
  Follow-up is `min(now, absolute_outcome_time)-enrollment`. Observed-outcome
  monitoring counts both binary 0 and 1. Its no-selector native getter rejects
  multiple endpoints; pending detection checks every outcome of every patient.
- Trial duration is checked strictly above the maximum, before the enrollment
  cap. Both are gated by trial minima. Completion adds final follow-up to the
  current candidate time, including its overshoot, and analyzes again. Early or
  no-active-arm stopping does not add that follow-up.
- Arm/group minima can reopen even permanently closed arms; group minima can
  override maximum-enrollment status. Start/duration checks then apply. Thus the
  ordering is part of the contract, not interchangeable validation.
- Whole-trial final selection permits statuses open and enrollment-maximum,
  uses strict final cutoff and resolves equal probabilities to the later arm.
  Group selection permits only open arms and has a distinct early-threshold
  override. A group result can subsequently be overwritten by whole-trial
  selection. Early-winner classification also treats completion differently.
- Posterior ranking includes all arms, including closed and suspended arms.
  `SetUpTrialInfo` (`0x9b524`) copies the complete trial arm map;
  `UpdateCalculator` (`0x9aafc`) visits every arm/group; `CompareAllTreatments`
  (`0x9d4d4`) pushes that same complete map into ranking vectors without status
  or eligibility queries. This is a source-path proof, not an additional numeric
  original-posterior fixture. Allocation and final selection filter statuses later.
- Native allocation uses power transformation followed by iterative proportional
  floor enforcement, not the existing mixture or clamp-and-renormalize policies.
  Forced equal run-in rounds its enrollment requirement up to an arm multiple
  and uses deficits across all arms; it bypasses the ordinary tuner. Inverse-CDF
  assignment uses `uniform <= cumulative`, choosing the earlier arm at equality.
- The ordinary native predictive-calculation vtable entry is a no-op. Dormant
  predictive-policy code with supplied scalars is not evidence for a recoverable
  predictive model, and the guide does not advertise that model.

Desktop constructors and missing-attribute XML defaults differ: desktop tuning
uses floor .1, forced equalization and stop-when-one-arm; missing XML attributes
instead produce floor 0 and false flags. A named native reader must preserve the
selected source profile rather than silently choose one default set.

The exposed binary scenario supplies `ProbSuccess` and `TimeBetweenEvals`.
Its native maximum-evaluation count defaults to one, making response delay
exactly `TimeBetweenEvals`. Latent XML can request multiple discrete evaluation
times; that is distinct from the exposed fixed-window desktop scenario.

The exposed TTE scenario has exactly one Mean or Median parameter, measured
in months. Native loading converts `mean = median/log(2)` when needed. Optional
`MaxTTE` is a latent input; without it the native loader uses DBL_MAX, and the
exposed desktop does not supply a finite censoring horizon. The TTE generator
records event indicator 1 when raw survival is at or below the horizon, 0
otherwise, and observation time `arrival + min(raw_survival,horizon)`. A
supplied native calendar must retain that indicator separately: a censored
record is observed for monitoring, while it is not a failure event for survival
inference. Existing Python event-time-only replay cannot silently stand in for
that general censored contract. Key native RVAs: constructor `0x12f9b0`,
parameter loader `0x133aa0`, value generator `0x1317b0`, time getter `0x131ec0`;
managed default scenario constructor `0x6f30`.

The additional sufficient-statistic fixtures execute original helper `0x1b62d0`
with the original patient visibility and follow-up functions. Ten fixtures cover
pending, event and censor cases, including exact observation-time ties. Censored
records contribute zero events and stopped follow-up; the native inverse-Gamma
update sums event indicators and follow-up separately.

## Executed evidence and limits

Read-only probes executed original x86 arithmetic and original managed IL using
explicit map/string/container/status services. Families cover allocation floor,
monitoring, visibility, pending detection, minimum criteria, arm constraints,
status transitions, group and whole-trial final selection, and dormant predictive
policy. Original posterior calculation and RNG were not substituted by invented
models: controller probes supplied posterior values and label that boundary.

Six integrated cases execute the original `GetTreatment` instructions and
recursively the original monitoring, constraints, status and final-selector IL:

| Case | Original result |
|---|---|
| Exponential monitoring, pending outcome | All arms temporarily closed; group then whole final selection; `Stop Trial`; no allocation |
| Exponential monitoring, no pending outcome | Reopen all; analyze; allocate (.6,.4) |
| Continuous monitoring | Analyze; check constraints; allocate (.6,.4) |
| Enrollment cap at candidate time 5, follow-up 2 | Analyze at 5 and again at 7; final arm 0; `Stop Trial` |
| Early-winner posterior (.96,.04) | Close other arm; group then whole selection; `Stop Trial` |
| Trial minimum not met | Skip posterior look; check constraints; allocate from cached values |

The allocation branch executes original weighted aggregation and native tuner;
its dispatch and return-string services are supplied. These cases do not claim
unmodified Windows-process, posterior or RNG equivalence. The portable [reference harness](../tools/arand_native_reference/README.md) now
regenerates all 98 fixtures across thirteen families. It requires a local original
library with the exact hash above, declares every runtime service, enforces
instruction/subprocess limits, and contains no original binary.
[Committed fixture manifest](fixtures/arand-native/manifest.json) records counts
and hashes. These source fixtures support the next Python workflow; they are
not yet tests of a completed native Python controller.

## Dependency order for the next implementation

1. **Done:** preserve the original-instruction runners as a portable, hash-pinned
   reference tool with declared runtime services and bounded instruction counts.
2. Build a named single-endpoint supplied-calendar profile around the recovered
   status/scheduler/allocation/final-selection contract. Retain the existing
   explicit-policy API. Validate each component and integrated recorded histories.
3. Connect independently validated binary and survival posterior kernels using
   the recovered all-arm ranking population; add integrated numerical histories
   and explicit TTE censor indicators.
4. Integrate the recovered binary response-delay and TTE scenario conventions;
   finish persisted inputs and report semantics before claiming a complete
   desktop simulation workflow.
5. Run independent numerical checks, installed examples, and the full regression
   suite on final code. Do not promote coverage merely because a new API exists.

Remaining native-specific gates are full posterior/simulator integration, report
aggregation/quantiles, wrapper RNG policy and missing-attribute XML closure-action
defaults. The exposed managed action defaults are known, so an explicitly named
desktop profile need not guess those XML defaults. Independent Python random
streams can be supported with that distinction stated.
