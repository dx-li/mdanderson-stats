# U-BOIN accelerated-titration audit

The source-defined Stage-I accelerated-titration option is exposed through
`simulate_uboin` and the deterministic `uboin_stage1_titration_plan` replay
helper. It is a dose-assignment prelude; it does not change the Dirichlet
posterior, utility calculation, admissibility rules, or ordinary U-BOIN stage
decisions.

## Primary source contract

The official U-BOIN V2.4.4.0 guide, section **Accelerated Titration**, says to
start with one patient at the configured starting dose and escalate one patient
per dose level until the first DLT, the second moderate (grade-2) toxicity, or
the configured titration cap. If a toxicity trigger occurs, or the cap is the
highest dose, add `m-1` patients at the last dose before resuming cohorts of
size `m`. If a lower cap is reached without either trigger, start cohorts of
size `m` at the next higher dose. The guide states that cohort size one and a
starting dose at the highest level make the option have no effect.

Source record:

- Official guide: [Titration.pdf](https://biostatistics.mdanderson.org/shinyapps/UBOIN/Titration.pdf)
- Cached source: `/Users/dxli2/math stats/mdanderson-stats/research/raw/UBOIN/Titration.pdf`
- Cached PDF SHA-256: `8063427ee0f3feaf3e0eb69bbc6ac93ae37c9fa40096df72faa3b9e5381611eb`
- Text extraction: `/Users/dxli2/math stats/mdanderson-stats/research/raw/UBOIN/Titration.txt`
- Application version: U-BOIN 2.4.4.0, PID 1014; source inventory is in
  `docs/uboin-sources.json`.

The paper identifies stage-I escalation, while the official titration guide
adds the optional singleton path and its cap/top-up rules. This implementation
uses the guide wording for the optional path and resumes the existing Python
controller only after the first cohort at the exit dose is complete. It does
not introduce an assessment delay: the source does not specify one, and this
simulator receives complete categorical outcomes.

## Input and handoff conventions

The toxicity categories are ordered and `dlt_level` is the existing split index
whose category and higher count as DLT. Because ordinary binary toxicity data
cannot distinguish grade 2 from DLT, accelerated titration requires an
explicit one-based `grade2_toxicity_level` identifying a category below that
split. For example, with categories `[none, grade 2, DLT]`, use
`dlt_level=2` and `grade2_toxicity_level=2`. The helper rejects overlapping
grade-2 and DLT categories rather than treating every non-DLT outcome as grade
2.

The dose cap and grade-2 category are explicit Python inputs. `None` for the cap
means the highest configured dose. The dose path, toxicity trigger, cumulative
grade-2 count, exit reason, top-up size and next ordinary dose are replayable
from one-based observed toxicity categories. The simulation applies the
design's hard `max_patients` bound to the `m-1` top-up; if a clean lower cap
exhausts the budget, the final decision is evaluated at the last treated dose.
The titration observations are retained in the ordinary joint-count ledger.

## Direct treatment-ledger references

The following examples use cohort size `m=3`, four doses and start dose 1:

| Observed titration path | Exit rule | First-cohort handoff |
|---|---|---|
| No trigger through dose 4 | Highest-dose cap | `[1, 2, 3, 4, 4, 4]` |
| DLT at dose 2 | First DLT | `[1, 2, 2, 2]` |
| Grade 2 at doses 1 and 3 | Second grade 2 | `[1, 2, 3, 3, 3]` |
| Clean cap at dose 2 | Lower cap, no trigger | `[1, 2, 3, 3, 3]` |
| DLT at lower cap dose 2 | First DLT | `[1, 2, 2, 2]` |

With start dose 3 and clean arrival at dose 4, the handoff is `[3, 4, 4, 4]`.
If the total-patient limit is 3 and DLT occurs at dose 2, the top-up is
truncated to produce `[1, 2, 2]`. The focused tests use these paths to verify
both the standalone replay contract and the simulator's joint-count ledger.
