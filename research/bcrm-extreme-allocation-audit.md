# bCRM extreme-dose allocation probability

The cached bCRM simulation guide V1.R4.M0, §3.2.4, p. 21, defines the
adjusted probability for an extreme dose as

```text
p = p_T ** (1 + gamma * (p_o - p_T))
```

Here `p_T` is the user-entered target allocation fraction, `p_o` is the
fraction of subjects already allocated at that dose, and `gamma` is the
correction factor. The guide says zero disables correction, positive values
increase its strength, and the resulting probability is constrained to
`[0.1, 0.5]`. This port requires the target fraction to be strictly inside
`(0, 1)` to avoid the source formula's undefined zero-base/negative-exponent
case; observed allocation fractions may be in `[0,1]`, and correction is a
finite nonnegative scalar. Log-threshold comparisons implement the formula
without an overflowing power.

The helper returns only this randomization probability. It does not implement
the bCRM controller, dose/endpoint logic, futility rule, bivariate posterior,
or native MCMC behavior.


Root integration passes all seven focused decision/allocation tests, including
a 50-digit Decimal reference. Four further cases span the smallest positive
float, a target immediately below one, and maximal finite correction, with
overflow, invalid arithmetic and divide-by-zero set to raise. All produce the
source-clipped probabilities. The combined checks used 0.163 seconds after
imports, 147.23 MiB peak RSS and no process swaps. Ruff/format checks passed;
targeted mypy passed in the implementation checkout. No new CI or dependency
was added. The guide example imports the helper from the public package.
