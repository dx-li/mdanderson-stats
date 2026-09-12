# BF-BOIN calendar simulation

`simulate_bf_boin` replays BF-BOIN with asynchronous enrollment and delayed
DLT assessment. The default inter-arrival gap is uniform on
`(0, 2/accrual_rate)`; exponential gaps are available with
`arrival_distribution="exponential"`. The first patient starts at calendar
time zero. A DLT endpoint is calibrated with a Weibull distribution satisfying
`F(window)=p` and `F(window/2)=p/2`; assessment is at the earlier of the DLT
time and the window. At `p=0` no DLT occurs and assessment is at the window;
at `p=1` the limiting DLT time is `window/2`.

Each escalation cohort is enrolled at its current dose. While its outcomes are
pending, subsequent arrivals may be assigned to the highest eligible lower dose
under the core BF-BOIN rules. Only observed assessments affect eligibility and
decisions. The `assigned` result counts every assignment, while `patients` and
`toxicities` are final evaluated counts after follow-up. Per-trial assignment
counts and assessment times are retained in `assigned_history` and
`final_assessments` for reproducibility checks.
`escalation_end` records the last escalation decision; `trial_duration` includes
final follow-up.

The simulator supports the primary escalation-time backfill mode. Expansion
after escalation and titration are separate workflows and are intentionally not
accepted as silently ignored options.
