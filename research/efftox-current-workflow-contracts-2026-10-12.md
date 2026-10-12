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
