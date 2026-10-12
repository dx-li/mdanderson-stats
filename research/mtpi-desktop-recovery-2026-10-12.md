# mTPI desktop recovery and scope

Catalog 72's current version 2.1 download contains `mTPI.xlsm`, `mTPI.xll`,
`mTPI-middle.R`, the monitoring script, manuscript and 2012 guide. The guide's
operational scope is monitoring-table generation and scenario simulation.
Original 2007 TPI multiplier tuning and informative-prior loss recalibration
are separate method extensions, not current spreadsheet inputs. Their documented
limits are retained; recovering this application does not invent those methods.

## Pinned external assets

Official archive: https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/TPI/TPI_V2.1PID439.zip

| Asset | SHA256 |
| --- | --- |
| Archive | `4e87f1a816bbf1ffb1a7817ac2a87d3de019090e78d4c24c07ec6c8b44059c8c` |
| mTPI.xlsm | `7bfa42bc88f1fe1aaf437d5b902770aef53590b488526ce5d320b410530ced19` |
| mTPI.xll | `c946b1bc48baafe1f6477767a6d4f64f97ca7ff95398c9b29ed9159f79d0c858` |
| mTPI-middle.R | `2629c28a1bf607329b125d7fca6715d8d1bfd0615162ebc4d091b2c603f5f860` |

All originals remain under ignored `research/raw/mtpi-2026-10-12/`. No original
source, binaries, document prose or macros are distributed. Reference drivers
hash-check externally supplied files; six synthetic numerical/cached fixtures
are retained under `tests/fixtures/mtpi-desktop-*`.

## Recovered executable contracts

The monitoring VBA overrides unsafe cells to DU using Beta(1,1) and strict .95;
otherwise it requires a strict three-way UPM winner. Ties remain blank. The
original cached table has 495 valid count cells. This is cached original output,
not new Excel execution.

Original decision instructions at XLL RVA `0x83d8..0x8789` execute with original
PE memory and host vectors; the beta service at `0x6080` is supplied by SciPy.
2,496 cases cover four targets, four current sample sizes, every toxicity count,
three current doses, four neighboring counts, and both carried indicator states.
They agree with unchanged R decisions for next dose, exclusion, indicator and stop.
This does not establish native beta-library or full Windows initialization parity.

Allocation uses Beta(1,1). The next-dose safety shape is genuinely
`Beta(y_next+.5,n_next+.5)`, omitting toxicity subtraction. Lowest-dose safety and
highest-dose underdosing use Beta(.5,.5); top underdosing stops above .8. The
exclusion marker can be overwritten, and the next-dose indicator is carried at
the highest dose. Starting there leaves the native slot uninitialized (initially
a temporary vector pointer); Python's explicit legacy profile rejects it.

Correction to the prior audit: the outer dose branches are nested and strict
UPM maxima are mutually exclusive. There is no established multiple-increment
per-cohort defect. The earlier unsupported claim has been removed.

432 complete unchanged-R trials cover four targets, cohort sizes 1/3, capacities
4/6/12, starts 1/2, three toxicity scenarios and three uniform tapes. Counts,
paths, selections and historical flags match. Capacity 4 with cohort 3 confirms
native whole-cohort overflow; safe Python clips its last cohort.

Final estimation uses Beta(.005,.005), inverse-variance PAVA and the treated/
excluded prefix. XLL rank instructions at `0x8a11..0x8af6`, supplied fitted means,
add zero-based `i*1e-10` from the second dose and choose the higher exact tie.
The R script uses one-based perturbations and the first tie. Its `2:1` indexing
bug is not reproduced. Separate APIs retain the two estimators. Native ranking
probes do not execute native PAVA; unchanged R and prior exact-rational PAVA
references support the fitted means independently.

Original MT19937 constructor `0x1aa30`, seed wrapper `0xf560` and draw wrapper
`0xf5c0` establish seed 2 once per full request and `(uint32+.5)/2**32` uniforms.
2,000 consecutive original values match NumPy's uint32 transformation exactly,
including multiple state regenerations. This is not NumPy's 53-bit random_sample.

The integrated public API reproduces all ten cached scenarios, 2,000 trials each,
consuming 577,344 uniforms. Every mean allocation and pooled toxicity rate matches
exactly; selection differences are at most 1.25e-16 from native no-dose subtraction.
Cached complete-workflow evidence plus bounded original instruction execution
is not a claim that every Excel/XLL instruction has executed.

## Integrated workflow and acceptance boundary

`mtpi_desktop.py` imports bounded numeric XLSM inputs without macros, separates
monitoring and simulation studies, and saves a versioned JSON schema.
`mtpi_desktop_trial.py` implements paper-default and explicit legacy replay and
XLL final ranking. `mtpi_desktop_simulation.py` connects every scenario, actual
stream consumption, selection/allocation MCSE and stopping/safety diagnostics.
`mtpi_desktop_report.py` produces standalone reports with captured study,
profile and seed. Reopening can reproduce execution settings as well as inputs.

The current desktop scope has no automatic prior/multiplier tuning operation.
Its users choose target/margins/scenarios and inspect operating characteristics.
The existing original-TPI and prior-sensitivity extension limitations therefore
remain separately documented, not silently erased from the repository.

Source compatibility, mathematical correctness and clinical validity are separate.
No source defect is a silent default. Full acceptance evidence and exact remaining
contracts belong in the checkpoint and validation manifest after verification.
