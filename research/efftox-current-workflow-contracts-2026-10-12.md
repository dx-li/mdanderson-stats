# Next EffTox current-version workflow contracts

This is a source checkpoint, not an implementation/status promotion. The current
5.2.3.0 archive was retrieved again and matches SHA256
`84fc159c00bb825d0e28c18d095a77a98a8fd97e17c8b2451f9e03e3484b5c47`.
The ignored extraction is `research/raw/efftox-2026-10-12/extracted`.
Do not use the old four-coefficient `SampleTrial.xml`/`SampleSim.xml` as if they
were the current binary model. `Docs/EffToxSim_Default.sim` has six prior means/
SDs, current ProgID `C893BCF6-6184-419a-8173-354C4BDE8728`, Model1, five physical
doses, current binary/trinary contour metadata and conditional efficacy truth.
The current GUI's binary-only restriction was already recovered separately.

## Newly executed current Lp utility evidence

`tools/reference_efftox_native_lp.py` executes original GetP (MethodDef129,
RVA0x7674), its complete original root-finder constructor/bisection, and GetUtility
(MethodDef128, RVA0x6568), using the existing hash-pinned original-IL VM. Standard
Log/Pow services and argument stores were added; all15 existing legacy fitter
fixtures regenerate unchanged. The new21 utility fixtures are in
`research/fixtures/efftox-native/lp-current.json`.

For a **supplied identical shape**, all21 utilities agree exactly with
`1 - (((1-e)/(1-e0))**p + (t/t0)**p)**(1/p)` in this environment.
Original RootFinder constructor RVA0x5318 sets tolerance0.0001 and at most20
iterations. Consequently, its fitted shapes are slightly different from the
more accurate Python root. For the standard (.5,.65,.7,.25) contour, native
p=0.9773412187692126 versus Python0.9773682674593841; the native middle-point
contour residual is1.85447e-5. Across these21 cases, utility differences using
each solver's own shape reach1.03021e-4. Preserve this distinction: do not claim
exact native decisions for near-ties, and do not degrade the scientific default
silently to the old numerical tolerance.

## Remaining source-recoverable contracts

- Current saved study: parse/preserve the six-coefficient current XML format and
  active binary settings. The conditional-truth checkbox is `ProbsAreDependent`;
  `TrueResGivenNoToxProbs` values remain saved when independent mode is selected.
  Resolve source default/migration handling explicitly; old Model0/four-parameter
  samples do not establish current model interpretation.
- Patient files: original EffTox2Data Patient.ToXML RVA0x12830 writes ID,
  BeginDate, DoseCalcedIndex, DoseAdminIndex and Outcome. Cohort.ToXML RVA0x9490
  groups Patient children. Patient IDs and dates are identifiers; the guide permits
  administered dose to differ from calculated dose. Missing either binary outcome
  means unset, not a nontoxic/nonresponding observation. Establish the outcome-code
  mapping and native parser roundtrip with original instruction references.
- Conduct: managed BBQEffToxTrialConductor.GetDecision RVA0x6ca4 in
  EffToxManagedWrapper.dll computes the trailing-cohort prefix before its first
  native call. It does not simply reuse the CRM controller. Partial-cohort handling
  depends on `m_allowEscalation`/`m_allowDeEscalation`; both false skips the second
  current-data calculation. The native conductor begins at RVA0x41c10 (thiscall).
  Recover pending lookahead, no-skipping, final selection, and the advertised
  GO/WAIT/STOP_EARLY/STOP_TRIALCOMPLETE decisions before integrating history APIs.
- Simulation/report: original help requires true efficacy/toxicity/utility,
  selection fraction, average treated patients, no-selection, all input settings,
  mean Rselect and prior-only probability/mean/utility diagnostics. Current help
  does not advertise the old sample's FirstCohortOutcomesOnly attribute. Client
  GenerateHTMLsimReport RVA0xc92c and OutputSimulationParameters RVA0xd030 expose
  report mapping. Native g_PerformSimulation is RVA0x11fc0 in the calculations DLL.
- Current prior-calibration input/output precedence: `NewestHyperParams` controls
  whether saved elicitation inputs require recalibration. Map it explicitly and
  retain independent numerical/calibration diagnostics. Do not silently invent a
  fitted prior or claim native numerical optimizer/RNG identity.

Historical inverse-quadratic utility/trinary behavior is a separate compatibility
scope from the currently advertised binary GUI. Any eventual catalog promotion
must declare its version/workflow scope and retain those historical boundaries,
not erase earlier evidence. No production EffTox changes or status promotion were
made in this source checkpoint.

### Patient mapping resolved during source inspection

Client `PatientDlg.GetPatientData` RVA0x2bd0, IL154..249 maps the radio choices
as follows: unset=-1, toxicity-only=0, neither=1, both efficacy/toxicity=2,
efficacy-only=3. This is **not** the usual two-bit numeric ordering.
`EffToxManagedWrapper.CopyPatientList` RVA0x50c0, IL87..157 converts component
response/toxicity states; if either is unknown (2), both native components are
set to unknown. Dates are retained as treatment identifiers but the help states
they do not enter the calculations. Patient.ToXML uses `DoseCalcedIndex` and
`DoseAdminIndex` separately, as the guide permits actual treatment to differ.

The managed conductor's trailing-cohort scan uses equality with cohort size,
not the CRM helper's greater-or-equal comparison. Its first native analysis uses
only the completed-cohort prefix; optional within-cohort escalation/de-escalation
then triggers a second analysis including the current cohort. Preserve this
separate contract rather than sharing CRM's safety/entry-replay controller.

## Further current-version recovery and saved-file implementation

`efftox_desktop.py` now reads the current six-coefficient XML, preserves unknown
metadata, maps all five native outcome codes, counts outcomes at the **administered**
dose, preserves inactive conditional truth, rejects impossible active joint truth,
and prevents `NewestHyperParams=input` from silently using stale output priors.
This is saved-file interchange, **not completed conduct/simulation integration**.
The original default study parses (five doses/five scenarios); synthetic tests
cover patient codes, round trips, invalid/ambiguous files and prior precedence.

Current client `CalculateDose` RVA0x7a80 sets **both** `m_allowDeEscalation` and
`m_allowEscalation` false (IL168/175). Thus the advertised current GUI uses the
completed-cohort prefix only; the optional second within-cohort analysis in the
managed wrapper is not activated by this client. Preserve that distinction from
latent wrapper capabilities. Patient outcome codes are independently confirmed
by this caller's switch at IL734 and its component assignments.

The managed wrapper's native BBQ vtable (FieldRVA0x85114) maps count collection,
pending consistency and selection to RVA0xd710,0xd220,0xd630. The latter creates
the BBQ analysis engine and calls RVA0xf790: posterior summaries at0xf540,
observed per-dose totals at0xf340, then selection at0xfc90.

### Executed original admissibility, not an inferred generic EffTox rule

`tools/reference_efftox_current_admissibility.py` executes **all original branch
instructions** at RVA0xecc0 of `EffToxManagedWrapper.dll`, SHA256
`6b266b3927bec98fd6d64f5767a1110cfccc3fe6bd18c7d438b9cfd538e4479c`.
Only the checked integer-vector accessor is a host service; inference is supplied
as tail probabilities. The250 fixtures in `fixtures/efftox-native/current-admissibility.json`
cover empty/endpoint/interior/noncontiguous tried-dose sets, starts0/2, all five
candidates, threshold equality and either side of the probability cutoffs.
Recovered rules (indices zero-based):

- Toxicity-tail probability **at least** its cutoff; equality passes.
- With any observations, candidate lies from `min(tried)-1` through `max(tried)+1`.
- Efficacy-tail probability must be **strictly greater** than its cutoff, except
  at candidate `max(tried)+1`, which receives the exploration exemption.
- With no observations, the admissibility helper admits only the starting dose
  after the toxicity test. However, the **selector** at0xfc90 bypasses the helper
  and returns the starting dose whenever that starting dose has zero observations.
- The selector computes Lp utility at posterior marginal means, uses strict
  improvement (first index wins utility ties), and reports stop when no candidate
  qualifies. It does not receive an interim/final flag. The outer conductor maps
  non-stopping decisions to trial-complete only after comparing patient count
  with maximum; final selection therefore uses the same native reachability rule.

These are **different** from the generic `efftox_decision` profile's strict
toxicity comparison, first-untried exploration and final unrestricted selection.
Do not silently change that existing scientific API. A named current-desktop
controller and original selection/pending probes are still needed.

Pending checking at0xed80 first analyzes the **observed-only baseline**, then
iterates completion count matrices (enumerator0x23200), requiring every completed
analysis to retain the baseline's dose and stop flag. Agreement among completed
analyses alone is insufficient. The outer controller0x41c10 maps inconsistency
to WAIT. This is static instruction evidence; full pending enumerator execution
and numerical inference integration are still outstanding.

### Executed selector and named Python profile

`reference_efftox_current_selection.py` now executes the complete original
selector0xfc90 and admissibility helper0xecc0 for250 supplied-summary cases.
The utility callback, checked vector access, memcpy and security-cookie return
are explicit host services; original selection branches execute unchanged.
`fixtures/efftox-native/current-selection.json` confirms unobserved-start bypass,
no-admissible stopping, reachability, threshold equality, exploration and
first-index utility ties. Combined with the250 admissibility cases, all500
fixtures pass against the named `efftox_desktop_decision` Python profile. An
additional fixed-parameter posterior test verifies that this profile applies
current-desktop final reachability without changing generic final selection.
The accurate Python Lp shape and independent posterior remain separate from
original numerical/RNG identity. Pending/cohort/full simulation integration is
still outstanding; no catalog promotion is made at this checkpoint.

The expanded selector cases include negative utilities and the original -DBL_MAX
initial utility sentinel. Equal-to-sentinel utilities do not select a dose. The
Python compatibility profile retains this strict-improvement boundary.

`reference_efftox_current_pending.py` additionally executes the original
pending-consistency loop0xed80 for21 explicitly supplied completion/decision
streams (`current-pending-loop.json`). The observed-only baseline comparison,
dose/stop-flag equality and early exit execute as original instructions. Runtime
containers, enumeration stream and posterior callbacks are supplied services;
**the native enumerator itself is not yet validated by this probe**. Cases where
all completed analyses agree with each other but differ from baseline correctly
return false. BBQ conductor constructor0x2b40 initializes its extra pending-gate
flag at offset8 to false, so that gate does not force unconditional WAIT in the
current client path.

### Additional original controller and simulation/report evidence

- `reference_efftox_current_cohorts.py` executes managed GetDecision IL12..140
  before model construction, producing33 original prefix/new-cohort cases,
  including overfull tails that prove equality (not >=) with cohort size.
- `reference_efftox_current_controller.py` executes native outer controller
  RVA0x41c10 for32 supplied-service cases. WAIT on inconsistent pending outcomes
  takes precedence over selection/cap; otherwise no acceptable dose is STOP_EARLY;
  otherwise patient count at or above maximum is STOP_TRIALCOMPLETE; otherwise GO.
  Validation, inference, count collection and descriptive strings are host services.
- Client maximum-patient validation accepts any integer1..200, not just cohort
  multiples (RVA0xa4bc). Its Add Patient path uses the returned status without a
  separate capacity check. Because managed conduct omits a partial last cohort,
  the nonmultiple-cap conduct case needs an explicit safe policy/source comparison;
  do not silently allow enrollment beyond capacity as a scientific default.
- Calculations DLL simulation driver0x8220 calls single-trial simulator0x53b50.
  The latter fits at each cohort boundary (patient-count modulo cohort size),
  enrolls patients individually until maximum, and **always calls the native
  decision engine again at0x54228 after the loop**, including a partial final
  cohort or early stop. This is static source evidence, not a new native RNG run.
- Driver0x8220 computes true scenario utility extrema, accumulates utility only
  for selected trials, and sums selection counts excluding the no-selection bin.
  Rselect arithmetic0x8d8e..0x8dca is now executed in eight fixtures by
  `reference_efftox_current_rselect.py`. It is conditional on selection:
  `(sum_selected_utility - selected_count*minimum) /
  (selected_count*(maximum-minimum))`. No selected trials returns0. Constant
  utility with selections can produce NaN or infinity from cancellation/division
  by zero; preserve those only in explicit compatibility evidence, and report
  mathematically undefined normalization explicitly in scientific output.

New fixtures are `current-cohort-prefix.json`, `current-controller.json` and
`current-rselect.json`. Together with prior current probes there are594 original
instruction cases (500 selection/admissibility,21 pending loop,33 cohort prefix,
32 outer status,8 Rselect). Inference/completion-stream substitutions remain
explicit; these are not594 independent posterior validations.

## Integrated current workflows and negative-slope discovery

The next implementation adds `efftox_desktop_controller.py`,
`efftox_desktop_inference.py`, `efftox_desktop_patients.py`,
`efftox_desktop_calibration.py`, `efftox_desktop_simulation.py` and
`efftox_desktop_report.py`. See [the public workflow guide](../docs/efftox-desktop.md).
These close the previously missing completed-prefix/pending integration, explicit
bounded inference service, patient add/edit/delete/clear, calibrated-prior
precedence, complete-outcome simulation and HTML reporting paths. They do not
promote the application or assert native numerical posterior identity.

The 86 original cohort/controller/pending cases now validate production control
flow. Independently enumerated individual pending outcomes agree exactly with
all aggregated sufficient-statistic completions tested. Bounds are checked before
inference. A pending disagreement releases no dose. The safe capacity profile
analyzes the full final history; the named legacy profile retains the original
nonmultiple-capacity defect, but the patient-add API independently forbids
excess accrual. Prior calibration cannot change an accruing trial's design.
Simulation always reanalyzes after the final/partial cohort or an early stop.

### Original posterior integrand, not merely a fitted-value comparison

`tools/reference_efftox_current_integrand.py` now executes **the complete original
wrapper integrand RVA0x20b50**, including native Gaussian density arithmetic.
Supplied host services are checked integer/double container access, exp, logistic
and integer power. All outcome-cell arithmetic, likelihood multiplication,
negative-slope branches, tail indicators and output weighting execute original
instructions. Explicit prior objects contain their means, SDs and Gaussian
normalization. No original DLL is redistributed. The wrapper SHA256 remains
`6b266b3927bec98fd6d64f5767a1110cfccc3fe6bd18c7d438b9cfd538e4479c`.

The 81 cases in `fixtures/efftox-native/current-integrand.json` cover negative,
zero and positive slopes; negative/zero/positive association; empty and two
nonempty histories; and three Gaussian priors. Independently calculated joint
probabilities and Gaussian likelihood weights agree. The installed Python model
also agrees, checked by `tools/check_efftox_current_integrand.py`, with maximum
relative posterior-weight error `6.871784649697795e-15`.

This resolves the **integrand's prior convention**: negative toxicity slopes have
ordinary, nonzero untruncated Gaussian density. The likelihood uses the raw
logistic toxicity at each dose. It does not resolve original adaptive integration
limits/tolerances or certify that every finite-precision native integration result
matches an independently sampled full posterior.

Crucially, RVA0x20cca..0x20ce6 substitutes first-dose toxicity into **summary**
means and toxicity-tail indicators whenever beta_T<0. At RVA0x20d94 the likelihood
still loads raw current-dose toxicity from its separate stack slot. This is a
legacy summary inconsistency, not evidence of a positive-truncated prior.
`tools/reference_efftox_current_negative_slope.py` independently isolates 48
original summary cases and 15 native normal-log-density cases. Fixtures are in
`current-negative-slope.json`. The current source set now totals **738 cases**
(594 previous +63 isolated summary/density +81 complete integrand).

The Python option `toxicity_summary_profile="desktop_legacy"` reproduces the
summary inconsistency without altering the likelihood; its default `"model"`
keeps dose-specific modeled probabilities. The actual slope-prior distribution is
an independent explicit `monotone_toxicity` choice. No generic scientific default
was silently changed. Calibration profile metadata prevents reopening a Python
calibration under a different slope distribution.

Current native `AutoScaleDoses` MethodDef130/RVA0x7e38 also confirms the existing
centered-log-dose convention: if the first physical dose is zero, add the second
physical dose to every dose before logging, then subtract the mean log dose.
This source inspection agrees with the already implemented stable Python helper;
it is not an additional executed fixture claim.

### Remaining acceptance boundaries

Finish all-scenario and distinct-seed batch aggregation/reporting convenience,
compare operating-characteristic precision under explicit profiles, and audit
current help against the final public API before promotion. In particular,
posterior MCMC diagnostics must remain visible; running a patient workflow alone
is not evidence of adequate mixing or clinical validity. The initial installed
original-default example completed two 39-patient trials at 1000 draws/500 warmup/
4 chains, with maximum split R-hat1.074 across its fits. A higher-precision check
is being recorded separately. Historical trinary/legacy-utility scope remains
separate. Do not relabel the other 35 partial rows as simple interface work.

### All-scenario/batch and saved-file acceptance follow-up

`efftox_desktop_batches.py` now runs every saved scenario using validated saved
repetitions/seed or explicit overrides, with whole-request preflight. Unequal-size
batches aggregate by trial count, and conditional Rselect uses only selected
trials. Repeated outcome seeds and incompatible saved designs/settings are
rejected. Reports retain every batch's actual settings, prior summaries and
posterior diagnostics. Seven focused tests independently check weighted summaries,
seed separation/replay, full preflight, invalid aggregation and real posterior
report plumbing.

The installed original-file exercise identified and fixed whitespace handling in
`TrialIsShutDown`: the original document stores a newline-padded zero, not the
literal string `"0"`. Numeric settings accepted by the reader are now interpreted
consistently by conduct and add-patient APIs; three regressions cover whitespace,
decimal and exponent spellings of zero.

Original client `m_btnAddPatient_Click` RVA0xbddc, IL512..553 switches Simulation
to TrialConduct after addition. `DeleteLastPatient` RVA0xc2b8, IL41..83 switches
back to Simulation when the final patient is deleted; delete-all calls that same
method. Python now follows these file-mode transitions. Empty saved studies can
remain editable without scenarios; actual simulation rejects the absence of
scenarios. These are source-inspected contracts, not extra original-instruction
fixture counts. The source set stays738.

The three installed precision runs (2x39patients each) used model summaries at
1000draws/500warmup/4chains, model summaries at3000/1500/4, and explicit legacy
summaries at1000/500/4. All three retained identical count histories and selected
zero-based dose3 in both trials. Rselect was0.8664555937167867. Maximum split
R-hat was1.07402055 at default precision and1.05227437 at higher precision;
final-fit maxima were1.04192806 and1.02153324 respectively. This supports the
observed workflow/precision stability, **not** an unqualified convergence or
operating-characteristic accuracy claim from two replications.
