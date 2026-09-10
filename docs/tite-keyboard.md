# TITE-Keyboard effective follow-up and interim decisions

Catalog entry **135**, [TITE-Keyboard](https://biostatistics.mdanderson.org/shinyapps/TITE-KEYBOARD/),
is partially implemented for its ESS calculator, informative trimester weights,
approximate posterior-key calculations and interim dose decisions with pending
outcomes. Calendar-time trial simulation, time-to-DLT scenario generators,
flowcharts and integrated protocol reports remain pending. Numerical effective-follow-up
boundaries and lookup tables are available.

The app snapshot identifies version 1.2.2.0, updated December 15, 2025. Its technical
PDFs and the authors' [methodological paper](https://arxiv.org/abs/1807.08393) are
pinned in [provenance](tite-keyboard-sources.json). This is an independent Python
implementation of their effective-likelihood method.

```python
from mdanderson_stats import (
    KeyboardDesign,
    tite_effective_sample_size,
    tite_keyboard_decision,
)

# App Guide: two known outcomes, three pending patients followed 30, 48, 75 days.
ess = tite_effective_sample_size(2, [30, 48, 75], window=90)
assert abs(float(ess.effective_sample_size) - 3.7) < 1e-14

result = tite_keyboard_decision(
    KeyboardDesign(),
    patients=[3, 3],
    toxicities=[0, 1],
    pending_followup=[[], [30, 15]],
    current_dose=2,
    window=90,
)
assert result.action == "suspend_pending"  # 2/3 pending exceeds the app's 50% default
assert result.next_dose is None
```

## Follow-up weights and ESS

`toxicity_followup_weights` evaluates the assumed conditional time-to-DLT CDF at
each follow-up time. The default uniform timing gives `followup/window`. With
`trimester_probabilities=[v1,v2,v3]`, the window is split into equal thirds and
conditional DLT timing is uniform within each third, with masses v1, v2, v3.
The masses are nonnegative and sum to one; they are **conditional timing
probabilities**, not marginal toxicity probabilities. Tiny rounding error in their
sum is normalized; materially incorrect sums raise an error.

`nonpending` includes both observed DLTs and completed DLT-free assessments.
Each contributes one, even if a DLT occurred before the full window elapsed.
Only pending, DLT-free-so-far patients receive partial weights. ESS is
`nonpending + sum(pending_weights)`. Thus an informative late-onset prior
[.1,.2,.7] gives ESS 2.97 for the same example whose uniform ESS is 3.7.

All times use the same units. The standalone weight/ESS calculator accepts times
from zero through the window endpoint and empty pending vectors. Its final axis
contains pending patients; preceding axes and `nonpending` broadcast across
scenarios. At most 200 patients per scenario are supported. Time rescaling leaves
weights unchanged. The interim decision API requires pending follow-up to be
**strictly shorter** than the window; assessments at the window endpoint must be
moved into the nonpending count.

## Approximate likelihood and decisions

For observed DLT count y, completed non-DLT count m and pending weights w_i, the
exact pending-outcome likelihood is proportional to
`p**y * (1-p)**m * product(1-w_i*p)`. The TITE-Keyboard method replaces each pending
factor with `(1-p)**w_i`, giving the working beta posterior
`Beta(y+1, effective_n-y+1)` under a uniform toxicity prior. **This is a likelihood
approximation**, not an exact treatment of censored toxicity data.

`tite_keyboard_decision` uses the supplied `KeyboardDesign` target interval, edge
convention and safety settings. Supply one pending follow-up vector per dose;
`patients` includes everyone enrolled, while `toxicities` counts only observed
DLTs. Pending patients cannot include known DLTs or exceed patients without DLT.
The returned `posterior` contains key masses and the ordinary proposed move at the
current dose. Effective sample sizes and pending counts are returned for every dose.

Safety follows the paper's conservative enrolled-count calculation:
`Pr(p>target | enrolled_n, observed_DLTs)`, using Beta(y+1,n-y+1), **not** the
smaller effective sample size used for key decisions. As in the complete-outcome
implementation, elimination requires at least three enrolled patients; exclusions
persist and include all higher doses. Pass prior `eliminated` masks forward.
Safety overrides suspension and precision stopping.

The remaining rules are applied in this order:

1. Suspend if the current pending fraction is strictly above
   `pending_fraction_limit` (default .5). Exactly 50% is allowed. The app supports
   configurable limits up to .65; `None` disables only this percentage gate.
2. If the actual next assignment would escalate, suspend until at least two
   patients at that dose have ascertained outcomes.
3. If the design's precision threshold is reached, return `stop_enrollment`.
4. Otherwise return the usual escalate/stay/deescalate assignment, respecting dose
   boundaries and prior exclusions.

Suspension actions are `suspend_pending` and `suspend_escalation`, both with
`next_dose=None`; neither means that the trial has selected an MTD. Recompute the
snapshot when more outcomes become known. `stop_enrollment` also requires following
pending patients before final analysis. Once all outcomes are known, use
`KeyboardDesign.select_mtd` with the complete integer counts and exclusion history.

## Source differences and validation

The paper's example has one observed DLT and two pending patients followed for
1/3 and 1/6 of the window. It yields effective n=1.5 and de-escalation. The later
app's default 50% suspension gate instead pauses enrollment in this snapshot.
Setting `pending_fraction_limit=None` reproduces the paper's decision while
retaining the two-ascertained-patient escalation requirement. Fully observed data
recover the standard Keyboard posterior; conduct still applies this extra
escalation requirement when fewer than two patients have been observed.

The older app Guide prints a greater-than-three criterion for extra safety;
Python uses the same at-least-three convention as the existing Keyboard safety
implementation. Adaptive timing weights discussed in the paper are not implemented;
the app's uniform and informative three-piece timing choices are supported.

Validation includes the published ESS example, hand-calculated informative
weights, time-unit invariance, an independent closed-form fractional-beta integral,
published rounded decision thresholds, complete-data reduction and explicit
safety/suspension checks. Existing Keyboard reference tests exercise the shared
posterior kernel after its extraction for fractional effective counts.

## Precomputed effective-follow-up boundaries

`tite_keyboard_boundaries(design, max_patients=30)` tabulates ordinary posterior
transitions by observed DLT count y. The coordinate is the effective **non-DLT**
count `m = effective_sample_size - y`, not total ESS and not raw follow-up time.
The result is independent of the assessment window and timing-weight choice.

```python
from mdanderson_stats import tite_keyboard_boundaries

boundaries = tite_keyboard_boundaries(KeyboardDesign(), max_patients=12)
# One observed DLT: stop de-escalating near m=1.8756; escalate near m=3.0749.
assert 1.87 < boundaries.stay_or_escalate[1, 1] < 1.88
assert 3.07 < boundaries.escalate[1, 1] < 3.08
assert boundaries.moves(1, 3.06) == 0
assert boundaries.moves(1, 3.08) == 1
```

Each transition array has shape `(max_patients+1, 2)` and is indexed directly by y.
Columns contain the last effective non-DLT count before the transition and the
first after it. Nonzero finite pairs are adjacent representable floating-point
values. A pair `[0,0]` means the new decision already applies at zero; `[NaN,NaN]`
means it is unreachable within the specified maximum effective total. The first
transition permits stay **or escalation**; it need not create a nonempty stay
interval when the target key is at an endpoint.

`moves(y,m)` broadcasts inputs and returns -1/de-escalate, 0/stay, or +1/escalate
without rounding y or m. It rejects negative effective non-DLT counts or totals
above the table's range. Use it instead of rounding thresholds printed for display.
The zero-DLT row can have tiny positive thresholds near machine precision due to
conservative tie handling of a uniform, no-information posterior; the conduct
rules separately restrict decisions before sufficient outcomes are observed.

For n enrolled patients with y DLTs and c pending outcomes, m is
`n-y-c + sum(pending_weights)`, constrained to `[n-y-c, n-y]`. Thus a transition
requires the corresponding summed pending weight after subtracting the completed
non-DLT count. Under uniform timing, multiply that required weight by the window
length to express it as total pending follow-up time.

The result also includes `enrolled_patients` (1 through the planned maximum),
`eliminate_min` and `lowest_stop_min`: the inclusive integer DLT thresholds for
overdose safety. A threshold of n+1 means it is impossible at enrollment n.
Posterior boundaries **do not override safety, suspension, dose-range or precision
rules**. Continue to use `tite_keyboard_decision` for a complete interim assignment.
The tables expose numerical thresholds; formatted flowcharts and integrated reports
remain pending.

Validation reproduces the published thresholds for one through four DLTs and
checks one-DLT likelihood crossings using an independent closed-form Beta(2,b)
survival function. Lookup results agree with posterior calculations across 10,000
feasible effective-count scenarios, including asymmetric and endpoint target keys.
Every finite nonzero bracket is checked at both adjacent endpoints, and follow-up
lookups agree with interim conduct when suspension and safety allow assignment.
