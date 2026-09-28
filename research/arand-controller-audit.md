# Adaptive Randomization controller: source gaps

The local version 5.2 guide (`research/raw/ARAND/UsersGuide.pdf`) was inspected
again on September 28, 2026. Its existence and the recovered native binary do
not establish a complete controller contract. The existing posterior kernels
are described in [arand.md](../docs/arand.md).

The guide specifies these rules:

- Page 6, section 3.1: maximum enrollment and/or duration, and minimum
  enrollment before trial stopping.
- Page 7, section 3.2: allocation proportional to the best-arm probability
  raised to a nonnegative tuning power. During initial equal randomization,
  best-arm probabilities are not computed until the specified count is reached.
- Pages 7–8, section 3.3: strict lower best-arm cutoff for reversible loser
  suspension; strict upper cutoffs for early and final winner selection.
  There is no default final winner if the cutoff is not met.
- Page 8, section 3.3(iv): when maximizing only, a strict lower cutoff on
  the posterior probability of exceeding a fixed parameter threshold makes
  futility permanent.
- Page 8, section 3.4: final follow-up contributes to completed-trial duration,
  but not early-stopped-trial duration.
- Pages 9–10: exponential arrival gaps, per-arm early/final winner and drop
  probabilities, and treated-patient means and central simulation intervals.

The guide does not give the allocation-floor transformation, explicitly settle
ranking after permanent arm removal, order simultaneous triggers, or resolve
a hard duration cap arriving before minimum enrollment. Reversible loser
reactivation suggests continued comparison with suspended arms, but that is an
inference rather than a separately verified native algorithm.

No controller implementation was added at this checkpoint. A future port must
resolve these details from additional primary evidence or expose the choices
as documented Python policies. It must not label such choices as verified
native behavior. Repeating the guide inspection alone will not close these gaps.
