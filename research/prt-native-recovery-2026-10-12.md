# PRT1.0 native recovery

Current status69 remains partial. Original assets are locally recovered under
ignored `research/raw/next-desktop-2026-10-12/PRT_V1.0/PRT` and are not redistributed.
Conduct SHA256: `2943f71ab27e300c1168b2d9f6a93815b75efe753b6af2b543c1370d077a1123`.
Simulation SHA256: `52903a7af2e75998bf2a8cbda6e2049e7fa25e0981cf9806317841da13a91b54`.

## Projection contract recovered

`tools/reference_prt_projection.py` executes conduct RVA36fb–434c, including
actual native matrix inversion961d, on supplied synthetic posterior risk draws.
Thirty-two cases cover2/3/4/6/10doses, independent and strongly correlated draws,
negative weighted averages and weighted averages above one. These are standalone
projection references, not posterior, RNG or full trial parity.

Native behavior is **maximum over lower block endpoints of minimum over upper
block endpoints**, followed by a lower floor at zero (4267–4296). The final
published paper uses the reverse min/max ordering in Section3, Step3:
[author-hosted paper](https://odin.mdacc.tmc.edu/~pfthall/main/Biostatistics_LateTox_2008.pdf).
The current Python paper API correctly retains that published ordering. With
negative inverse-covariance weights the two orderings need not agree: one
synthetic10-dose case differs by0.04992356. Do not silently substitute the
native ordering into the paper API.

The lower floor does not establish probability validity. Four native cases still
return values above1.005 (maximum1.01016399). The paper API currently rejects the
unbounded transformation in these cases, which is safer than feeding invalid
probabilities to predictive calculations. Native compatibility must be an explicit
profile with diagnostics; a mathematically valid alternative would require its
own declared contract and validation, not an unlabelled clipping change.

Independent NumPy covariance/block solves reproduce the native max-min/lower-floor
outputs to4e-13 on this grid. The fixture is portable synthetic data; no original
executable content is included.

## Simulation generator recovery in progress

The simulation executable's b786 helper evaluates a beta density in log space
using alpha/beta and supplied interior x. Its calibration caller at24f8–274d
evaluates100 locations stepped by1/101, divides by a searched scale, and matches
`1-exp(-integrated hazard)` to the configured toxicity probability within a
native single-precision1e-5 tolerance. Exact time-grid endpoints and subsequent
inverse-survival generation still need executable fixtures before implementation.
Do not substitute a simple beta event-time distribution: the observed code uses
beta-shaped hazard rates. Scratch disassembly/probes live under`/tmp/prt-*`.

Remaining scope: explicit projection compatibility/validity policy; exact native
patient-input conversion; calibrated hazard/event-time generator and accrual/
suspension timing; full saved study, conduct/simulation/report workflows and
independent acceptance. No application promotion from these source findings.

`tools/reference_prt_hazard.py` now executes the actual grid, calibration and
event-time generation blocks.24 calibrations and288 supplied-variate event cases
are portable. The grid is0 through t* in98 intervals, with the last endpoint
repeated; density locations are1/101 through100/101. Native calibration integrates
only the first98 rates and searches scale starting at1 with step10, reducing the
step by10 after overshooting, within a float32(1e-5) probability tolerance.
Independent SciPy beta densities and scalar search reproduce this grid.

Generation has two further compatibility defects: category selection uses one
uniform, a second uniform is drawn but discarded, and the first is reused within
the chosen category. The inversion also uses the next rate relative to the
corresponding CDF interval. All288 event transforms independently reproduce these
arithmetic choices. Tail event times may be extremely large; calendar conversion
must distinguish events beyond the assessment window from within-window events.
No Python simulation default is changed based on these defects.

Arrival code2e8c–2ed3 draws800 uniforms on[0,800*mean_interarrival] and sorts them;
it does not sample independent exponential inter-arrivals. This has been located
in source but still requires executable fixtures and calendar/controller recovery.

Six actual arrival generation/sort fixtures now verify all800 calendar points per
job, across means1/7.5/45 and two supplied random tapes. Another88 count fixtures
execute conduct223d–2441 with patient triples: third-column1 is censoring, not
an event. A censored record at intervalC is fully assessed; an event at any
interval1..C is fully assessed. All assignments count even when observed interval
is zero. The guide's C therefore counts assessed intervals in these inputs;
do not silently treat its column as the paper's current Y category.
Independent count/arrival checks pass94 cases. The arrival and hazard files still
do not establish exact native suspension/reconsideration or full trial parity.

Native conduct posterior pilots now execute213b–434c, including fixed source
seeds,10000 Gibbs iterations with5000burn-in, raw risk retention, covariance
transformation and lower safeguarding. Three synthetic histories cover two-dose/
three-interval moderate priors and six-dose/six-interval guide-sized diffuse priors.
All three pilot projections remain in[0,1]; this does not erase the independent
counterexamples above. The extra two seed-looking lines in the shipped conduct
settings file are not read: source214f/2159 sets seeds76849/13456 directly.

Two reduced posterior cases are independently checked by100/160-node tensor
Gauss-Hermite integration per interval, using the random-walk prior and count
likelihood. Quadrature changes are below2.3e-15. Both native5000-draw means and
independent Python four-chain4000-draw means lie within six respective batch
MCSEs of the integrals; maximum Python R-hat1.0006571. The native six-dose pilot's
batch errors are materially larger and are reported rather than presented as
precision-qualified clinical inference. See posterior fixture, native generator,
`tools/check_prt_native_posterior.py` and the dated quality JSON.

## As-of conversion and final-clock recovery

`reference_prt_calendar.py` executes native simulation RVAs3cae–3f2f and
3b01–3cae;64 synthetic boundary cases and3 final clocks are preserved in
`prt-desktop-calendar.json`. Independent arithmetic checks pass all67 cases.
An event is seen when its delay is at most the window and its event clock has
arrived. Native daily index is floor(observed time) below the window, window+1
above it, but zero at exactly the window (also below day1). Coarse interval uses
right-sided endpoint search plus the event indicator. Thus exactly-window
censoring loses follow-up and an exactly-window event enters interval1. Events
before day1 still enter interval1; they are not discarded.

Native final time considers only the last three patients. Synthetic earlier
pending patients demonstrate a final time4 versus complete follow-up90, and11
versus94. These are legacy defects, not acceptable silent scientific defaults.
This probe does not close the full suspension/controller contract. Current
paper-calendar behavior remains unchanged; PRT69 remains partial.

The same last-three-patient clock is independently executed through both native
finalization paths (3919 and3b01). A separate calendar-control fixture executes
3acb–3b01 and4000–4045 on90 synthetic states. Suspension advances exactly one
candidate-arrival index when enrollment is below the cap, sets its calendar
clock, and leaves enrollment unchanged. At the cap this block leaves the clock
unchanged before the final-clock block. Only a current cohort of three shrinks
to the remaining sample size; cohorts of one or two do not change in this block.
These are bookkeeping references, not recovery of the statistical suspension
predicate, native candidate-array exhaustion or the entire loop. The complete
calendar reference suite now passes157 tests. Full controller integration remains
open and must not infer untested rules from these bounded blocks.

## Native final selector and conduct boundaries

Final selection97e1–98fe is now executed for144 synthetic summary/count states.
A dose is eligible only with at least three assessed patients and posterior
exceedance strictly below the upper cutoff. Lowest-dose exceedance strictly above
the cutoff stops. The minimum distance is initialized from dose1 even when it is
ineligible; if no eligible dose matches that minimum the current dose survives.
Exact ties overwrite in ascending order, selecting the last eligible tie. An
ineligible current dose can survive this fallback. The paper selector remains
separate; do not introduce this defect into a scientific default.

Conduct4fc8–5552 has126 portable probes for acceptable/excessive categories using
supplied predictive criteria. Current-dose exceedance at the upper cutoff triggers
de-escalation, whereas lowest-dose stopping uses strict greater-than. With pending
patients in the acceptable category the native code compares PE with1-epsilon,
not the published epsilon. The no-pending code is5.11. These probes bypass the
predictive estimator; negligible-category branches and that estimator still need
separate recovery. No full native controller or trial parity is claimed.

## Saved input integration

`PRTStudy`, `PRTPatient`, `import_prt_conduct`, `import_prt_simulation`,
`read_prt_study` and `prt_study_counts` now implement bounded immutable studies,
strict version1 JSON, original labelled simulation sections, explicit discarded
unused seed lines, native censoring semantics and disjoint likelihood/pending
counts. Both original bundled inputs round-trip. All88 native count cases agree.
Simulation studies accept a final partial cohort,3..500 patients and1..100000
repetitions; C3..10 and D2..10 follow the desktop guides. This module supplies
inputs, not completion of simulation/conduct execution. PRT remains partial.

Additional conduct predictive probes execute4d3d–4fc8 (48 cases), and the complete
negligible-category predictive/controller block46ab–5552 (108 cases), from supplied
proper beta moments and interior patient risks. Independent enumeration reproduces
all probabilities within1.2e-16. Native categories use strict beta exceedance
comparisons (<lower and >upper). This confirms the downstream fixed-patient-risk
product enumeration, not how upstream posterior patient-risk inputs are formed.

The negligible branch hard-codes dose10 as the top dose, even for smaller designs.
Its late tests use PN against epsilon and next-dose PE against1-epsilon in several
pending branches. A lower-dimensional top dose can therefore receive an escalation
code. All156 independent probability/controller checks pass. Native codes retain
single-precision constant rounding. Boundary0/1 risks trigger native jitter paths,
which these interior-risk probes deliberately do not claim to validate.

Eighteen additional executable cases3493–35d7 resolve the upstream patient risks:
these are posterior means of **raw, unprojected** remaining-interval toxicity risks.
Only censored patients with observed interval belowC enter, preserving input order
within each dose. Two supplied coefficient draws are accumulated before averaging.
Independent accurate normal-CDF products agree within3.4975e-7 on the deliberately
nonmonotone grid; native normal-CDF approximation explains this recorded tolerance,
which is not a global error bound. The native predictive calculation therefore
multiplies patient posterior-mean risks. The published Python API instead averages
products of aligned projected posterior draws and preserves posterior dependence.
Those are materially distinct estimators, not merely different random streams.

Simulation late-allocation8ff1–97e1 has960 supplied-criteria cases, independently
checked. Unlike conduct, it uses the configured top dose. The same complemented
predictive thresholds remain. De-escalation selects the nearest-to-target lower
dose, not necessarily the highest non-excessive lower dose, with dose1 minimum
initialization and last matching distance. This probe supplies decreasing distances;
more arbitrary distance/tie grids remain necessary before integrating that branch.

Simulation pre-inference4674–47fe has208 count-summary probes. It suspends at ten
or more pending patients at the current dose; also with exactly three assigned and
no patients observed beyond interval0, or with at most three assigned to dose1 and
the first patient's interval still0. If at least three are assigned at the current
dose and none has observed interval>0, a later branch skips inference (suspending
at maximum enrollment). `observed` here is **any positive observed interval**, not
an event count; assessed and event counts are separate. These abstract count probes
exercise branch arithmetic, including combinations not reachable from actual
patient histories; they do not themselves establish complete trial parity.

Sixteen endpoint-risk probes supply fixed uniforms to the actual RNG call while
executing the complete negligible predictive/controller block. Native current-dose
no-event mass at risk1 is1-u*float32(1e-5), while event mass remains1; at risk0,
event mass becomesu*float32(1e-5) without reducing no-event mass1. Next-dose
no-event mass at risk1 remains1 and event mass1; risk0 event jitter usesfloat32(1e-7).
Thus predictive sums are not normalized and can approach4 for two certain-event
patients. Independent product-mass checks reproduce every result. This is a
material validity defect, not ordinary Monte Carlo error. Legacy diagnostics may
expose it, but valid scientific decisions must reject these invalid criteria or
use the explicitly corrected estimator. No silent normalization is justified.


## Integrated arithmetic/controller checkpoint

The explicit `prt_legacy` module passes2,867 tests against independent executable
fixtures and input-validity checks. The simulation branch fixture now has963
cases, including arbitrary distance/tie checks. Three further portable probes:

- moments:72 cases (60 supplied moments plus12 projected-draw summaries) confirm
  population-N SD, strict target exceedance, beta formed before alpha fallback,
  and fallback only for strictly negative shapes. Zero/NaN/infinity persist.
- controller:1,368 cases execute the full predictive/controller block, including
  untried next doses and exact next-upper-cutoff equality.
- output dispatch:17 codes execute original branch instructions while skipping
  formatted-I/O calls. Code6.0 reaches no decision-text branch. This is dispatch
  evidence, not formatting parity.

Current dose in conduct is the last patient record's dose. Rule4 does not specify
an exact lower dose. Do not import simulation's dose selector into conduct.
Source executable hashes remain unchanged. The three generators are
`tools/reference_prt_{moments,controller,output_dispatch}.py`; fixtures are under
`tests/fixtures/prt-desktop-*.json`. Native output defects remain named legacy
behavior, never silently corrected scientific defaults.

Reinspection of paper Sections3 and4 confirms isotonic conditional risks and
posterior-integrated products for pending outcomes. Existing paper APIs retain
that contract. Native projects total risk only and multiplies marginal raw-risk
means. A four-chain4,000-draw/2,000-warmup guide pilot (seed20261012) has maximum
betaRhat1.00035 and raw-riskRhat1.00050 but paper projection minimum-0.010342.
No silent clipping or profile substitution is supported by this evidence.
Saved-conduct execution, quality holds and full calendar/batch/report integration
remain required before entry69 can be promoted.

## Combined enrollment/calendar traces

The19th native fixture `prt-desktop-enrollment.json` adds18 supplied-delay paths
(72 trace steps). `reference_prt_enrollment.py` runs original instructions36ab–3f2f
and4000–4045, skipping only the already separately checked event-delay generation
and supplying explicit doses/delays. An independent Fraction/integer calendar
check reproduces every record. First analysis occurs at the **next candidate's**
arrival after the enrolled cohort, not at its last enrollment. Suspension skips
that candidate before evaluating at the following arrival. Resumption enrolls
from the then-current candidate index. Final cohorts shrink3 to1/2 as required.
The final clock still considers only the last three enrolled patients.

Precision audit correction: the first probes forced x87 control word0x37f
(64-bit significand), which exposed a boundary difference at arrival7+delay.1.
That is not the desktop startup precision. Tracing the actual CRT initializer
finds the `_controlfp(0x10000,0x30000)` call; original instructions in both
executables set0x23f (53-bit significand), independently of initial24/53/64-bit
precision. The Fortran initializer preserves those precision bits. Six direct
initializer probes are in `prt-desktop-fpu.json`. All reference generators are
corrected to startup precision. Prior37f fixtures remain available in git history,
not as default desktop claims.

The additional `prt-desktop-calendar-precision.json` has111 native boundary cases.
A separate integer/Fraction rounding implementation and a binary64 calendar
oracle validate these and72 enrollment steps. Startup precision observes the
arrival7/delay.1 event at stored clock7.1. Eight of20 non-initializer reference
files changed numerically after correction; every5,044 installed PRT test still
passes, so no production arithmetic change was necessary. The correction affects
reference provenance and exact timing claims, not the demonstrated controller,
projection, moment-fallback, endpoint-jitter or last-three-follow-up defects.

This evidence composes enrollment and as-of conversion, not posterior/controller
or whole-trial execution. The saved-conduct implementation and its frozen769-file
acceptance snapshot do not include a generated native simulation runner yet.
