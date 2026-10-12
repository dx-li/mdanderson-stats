# Native postprocessing contracts and integer-truncation defect

This is research evidence, not a change to production statistical summaries.
`tools/reference_wfmm_postprocess.py` pins the same WFMM1/2/3 checksums and archive
provenance as the boundary audit, supplies deterministic synthetic beta draws,
and checks native WFMM3 results independently of any production Python summary.
The adjacent JSON stores those supplied draws, settings and numerical outputs.
Original binaries remain outside artifacts. Native sampler draws are discarded.

Two cases use one fixed effect, 16 time points, 40 supplied coefficient draws
and one-level periodic Haar synthesis: an uncontrasted case and one with two
effect contrasts plus two time regions. Independent Haar formulas reproduce
saved beta draws. Sample means, sample SDs, absolute effect exceedance and
sign-tail probabilities match independent calculations. The native report keeps
`Lbeta_*` summaries before time contrast and separate `LbetaLt_*` regional
summaries. Existing Python `wfmm_summarize` returns the requested transformed
summary directly; both views can be computed explicitly.

For probabilities .1, .5 and .9 with 40 draws, native pointwise quantiles select
zero-based ranks n*p-1 and round through float32. This observation does not yet
establish the complete interpolation rule for noninteger ranks. Python's
existing documented linear-quantile convention is unchanged.

The original simultaneous-band calculation has an integer-absolute-value defect:
`ComputeZscores`, address 0x466d70 in the pinned WFMM3, converts each real
`draw - mean` to a signed integer at 0x466e0a, then takes integer absolute value
and divides by the real SD. `ComputeQuantilesAndFindSimbas`, address 0x4673a0,
also truncates `mean / SD` at 0x46757a before integer absolute value. Thus neither
operation is the intended real-valued standardized absolute deviation. This can
materially change simultaneous intervals and SimBaS, especially for small-scale
regional averages. It must not become the scientific default.

For these supplied cases, independent reconstruction of the native behavior
matches both bounds and SimBaS. Bands use Hazen interpolation, with index n*p-.5
(the source's 0.5 constant is at 0x4d05b0). SimBaS searches a quantile grid of
.01,.02,...,.99,.991,...,.999 and reports one minus the grid probability just
before the first threshold exceeding the integer-truncated mean/SD. The source
builds this grid at 0x467c0f–0x467cc1. It therefore has grid endpoints .99 and
.001 rather than an exact empirical tail over draws.

Python currently retains real-valued maximum deviations, empirical tail SimBaS,
and its declared linear quantiles. Those mathematical definitions already have
independent tests. This audit records a source contradiction rather than
silently changing the Python definition or claiming identical native reports.
It does not establish behavior for zero SDs, noninteger quantile ranks, multiple
native input-chain files, original random-effect reconstruction files or native
prediction-file contracts. Those are next targeted probes. Any future explicit
compatibility profile must be qualified separately from scientific correctness.
