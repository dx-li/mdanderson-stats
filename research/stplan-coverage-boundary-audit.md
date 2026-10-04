# STPLAN inverse-planning coverage boundary

## Proportional K-group totals

The native K-sample method defines a continuous proportional-total calculation,
not an integer enrollment-allocation procedure. The STPLAN 4.5 user manual,
chapter 11.2, printed pages 79–80 (`research/raw/STPLAN/user.txt`, around lines
2694–2702), says the original unequal group sizes establish proportions of the
total and computed group sizes remain proportional to those inputs. The methods
manual, §2.8, printed pages 12–13 (`research/raw/STPLAN/methods.txt`, around
lines 491–524), defines `s_i=n_i/n` and noncentrality `lambda=w*n`.

The cached source `research/raw/STPLAN/source/stplan-4.5/SOURCE/qbink.f90`
confirms the contract. Its comments at lines 50–52 describe the total as relative
group sizes when total size is being computed. `calc_number_in_each_group`
(lines 212–245) calculates `n=lambda/w` and assigns `grpsz(i)=n*prop(i)` using
double precision. It has no integer conversion, rounding, remainder assignment,
or tie rule. Python's proportional calculation and fractional output therefore
implement the source-defined method. Rejecting `integer=True` for a whole-vector
total is appropriate; adding an integer allocation would require a separately
specified Python policy and forward-power check, not a native method claim.

## Inverse bounds and branches

The methods manual §1 (`research/raw/STPLAN/methods.txt`, lines 56–60) describes
inverse planning generally as finding a zero of a monotone power function, while
noting that some parameters can be solved analytically. It does not prescribe a
single universal search bracket or root-selection criterion. Native routines
do provide per-method input ranges and sometimes directional branches. For
example, `research/raw/STPLAN/source/stplan-4.5/SOURCE/qnor1.f` documents that
solving the null mean constrains it to `<= XMH`, while solving the alternative
mean constrains it to `>= XML` (lines 24–33). The routine defines numerical
search constants `big=1e10`, `small=1e-10`, `srange=1e-8`, and
`brange=.99999999` (lines 99–103), then applies per-parameter ranges for SD,
sample size, significance and power through `qrange` (lines 165–197).

`stplan_solve` requires explicit bounds and solves only within that bracket,
then checks achieved power. This supports the source-defined forward methods
without pretending to discover all roots or a globally preferred design. The
remaining gap is native per-method automatic bound and branch behavior, plus
native session/report workflow; it is a compatibility/interface difference,
not an omitted inverse-power formula. No one-size-fits-all default should be
inferred from one routine's ranges.

## Classification

This source audit identified no missing statistical calculation. The subsequent
[saved-study workflow](stplan-study-workflow-audit.md) provides portable study
specifications, replay and JSON/HTML results, completing STPLAN's community
Python workflow. The catalog now marks this scope implemented. Native automatic
search defaults and exact session/report formats remain documented compatibility
differences; integer proportional allocation is not a source-defined procedure.
