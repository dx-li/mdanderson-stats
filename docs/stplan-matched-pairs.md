# STPLAN archived matched-pairs power

STPLAN 4.5 contains an older matched-pairs approximation that is commented
out of the current main menu. This Python API exposes its calculation; it does
not claim that the archived option remains part of the native menu.

For a preliminary paired 2×2 table in source order `(Z11, Z10, Z01, Z00)`,
`Z11` counts positive responses in both members of a pair, `Z10` positive in
group 1 and negative in group 0, `Z01` negative in group 1 and positive in
group 0, and `Z00` negative in both. The source's signed marginal difference
is proportional to `Z10-Z01`; this API uses a nonnegative `difference` in
that specified direction, as the source inverse does. The archived `PSIMPD`
routine computes `a=Z10+Z01`, `b=Z10-Z01`,
`N=Z11+Z10+Z01+Z00`, `r=(a+delta*b)/(2N)`, and
`psi=r+sqrt(r²-delta*(b-delta*(Z11+Z00))/N)`. The implementation evaluates an
algebraically equivalent normalized/hypot form to reduce overflow and
cancellation. `PMPD` then uses

```text
q = 1 - (delta/psi)^2 * (3 + psi) / 4
power = Phi((Phi^-1(alpha/sides) + delta*sqrt(n/psi)) / sqrt(q))
```

where `sides` is 1 or 2. As in the source, the two-sided option uses alpha/2
but reports power only in the specified direction of truth; it does not sum
both tails. The method is an approximate normal calculation, not an exact
McNemar test. The paired pilot table is essential because it captures
discordance; two marginal response rates alone do not identify it.

Inputs allow nonnegative table cells with total at least one, a difference in
`[0,1)`, sample size at least one, and a positive representable `psi`. The
shared STPLAN alpha convention requires `0<alpha<0.5`; the effective tail
`alpha/sides` must remain representable. Arrays broadcast up to the existing
200,000-case STPLAN limit. The Python implementation accepts finite table
weights above the archive's `1e10` per-input range when its normalized
calculation remains representable; this is a numerical extension, not a
claim of native-range parity.

```python
from mdanderson_stats import stplan_matched_pairs_power, stplan_solve

power = stplan_matched_pairs_power(
    difference=0.15,
    sample_size=700,
    z11=100,
    z10=10,
    z01=30,
    z00=80,
    alpha=0.05,
    sides=2,
)

plan = stplan_solve(
    "stplan_matched_pairs_power",
    compute="sample_size",
    target_power=0.8,
    bounds=(2, 5000),
    parameters={
        "difference": 0.15,
        "z11": 100,
        "z10": 10,
        "z01": 30,
        "z00": 80,
    },
)
```

`stplan_solve` can invert `difference`, `sample_size`, or `alpha` while keeping
the pilot table fixed. Choose bounds that stay within the forward method's
valid domain; after a solve, the shared inverse planner re-evaluates power and
rejects a result that does not attain the target. `sample_size` is continuous
by default because the original inverse gives a continuous planning value.
Use `integer=True` in `stplan_solve` when an integer pair count is needed; the
planner then searches for the smallest attaining integer within the bounds.

## No-pilot source modes

The archived interface also offers two initial-size heuristics. Supply both
`theta1` and `theta2` to use the source's group-proportion approximation
`psi=theta1+theta2-2*theta1*theta2`; its suggested preliminary pilot is `n/4`.
If both are omitted, the source uses `theta1=.9`, `theta2=.1`, calls this
“ultraconservative,” and suggests a pilot of `n/6`. That label is the source's
description, not a guarantee for arbitrary dependence between paired
responses. Neither pilot fraction is a total sample-size guarantee.

```python
from mdanderson_stats import stplan_matched_pairs_initial_size

estimate = stplan_matched_pairs_initial_size(
    difference=0.15,
    target_power=0.8,
    alpha=0.05,
    sides=1,
)
print(estimate.sample_size, estimate.recommended_preliminary_size)
```

The no-pilot closed form is accepted only when its positive square-root branch
exists and a forward calculation recovers the requested power. Its fractional
recommendation is preserved as a planning quantity. The current package API
does not reproduce the interactive menu, native table-of-values mode, or
native root-search/status messages.
