# TITE-BOIN12 calendar replay audit

The TITE-BOIN12 paper describes continuous patient accrual and interim
decisions as endpoint information becomes available. Section 2.2.3 applies
the BOIN12 movement/safety/utility rules to pending-data summaries; the BDA
route separately uses its posterior imputation model. The paper's Table 2
illustration supplies staggered patient times and endpoint event times, but
does not define a general reusable arrival process, look scheduler, queue
policy, decision lag, or exact handling of every simultaneous calendar event.

`tite_boin12_calendar.py` therefore exposes an explicit Python replay policy.
It accepts one interarrival gap and two endpoint-delay rows per possible
patient. The first cohort starts at the initial dose without an interim look;
patients within a cohort keep one dose and arrive at their staggered times.
At each cohort boundary, the gap from the previous patient's actual arrival
sets the candidate conduct-look time. The next cohort enrolls after the
configured decision lag. A strict pending gate suspends new cohorts until the
earliest pending endpoint event or assessment completion; each such update is
analyzed before release. Delayed cohorts do not queue, and subsequent gaps
start from actual arrivals. Same-time events and assessment deadlines are
visible to that look. A finite delay denotes an event within its endpoint
window; positive infinity denotes no event, observed at the window end.

After accrual stops, endpoints are followed only to the event or assessment
deadline. The complete-data TITE-BOIN12 selector then determines final OBD
diagnostics. A safety stop intentionally reports no selected OBD even if a
separate complete-data diagnostic calculation yields one. For BDA, the caller
must supply the four-cell Dirichlet prior and NumPy generator/sampler settings;
the replay stores dose-level summaries and discards each fit's draw arrays.

These timing and terminal-accounting conventions are explicit Python choices,
not native calendar defaults. In particular, the cached author simulation
orders toxicity and efficacy confirmation times separately and schedules the
next look at their configured order statistics together with the accrual
candidate time. This replay instead retries at each earliest unresolved
endpoint ascertainment, making intermediate updates visible. The source
supports the underlying AL/BDA decisions, suspension rule, and complete-data
selection; the replay does not claim exact reproduction of the paper's
adaptive Figure 1. The cached
Table 2 patient tape and observation ledger are suitable for checking the
staggered arrivals, event visibility, pending counts, and final follow-up
independently of an adaptive decision sequence.

Primary reference: [Zhou et al., TITE-BOIN12](https://pmc.ncbi.nlm.nih.gov/articles/PMC9199061/).
