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
