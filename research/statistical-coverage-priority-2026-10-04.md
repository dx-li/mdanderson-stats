# Statistical coverage priority review

This bounded review follows the request to prioritize Python statistical
capabilities for community use. It distinguishes absent calculations from
native application compatibility and unresolved source conventions. It is
not a census of every catalog entry, and does not promote any program's
catalog status or claim full site coverage.

The review used the existing source audits and guides below. No new native
execution or live application inspection was performed for this review.

| Program | Remaining issue in the inspected evidence | Implementation consequence |
| --- | --- | --- |
| BayesESS (#154) | Unknown-mean variance ESS has conflicting Hessian signs, an unspecified prior-df adjustment, and native scale/argument inconsistencies. | A real method gap, but its source contract needs resolution before claiming a port. See [variance ESS audit](normal-variance-ess-audit.md). |
| BaCIS (#153) | The published fixed cutoff and reported operating characteristics conflict; no automatic calibration algorithm was recovered. | Do not invent a native calibration procedure. Existing simulation accepts explicit cutoffs. See [simulation guide](../docs/bacis-simulation.md). |
| MERIT (#160) | Source describes interim tail cutoffs and isotonic pooling but does not resolve how previously stopped arms enter later pooling. | Retain the explicit Python policy and its limitation; exact native policy remains unknown. See [interim audit](merit-interim-search-audit.md). |
| CondiS (#157) | All eight refinement learners are implemented. The separate vignette regression example uses target-derived inputs and pre-split imputation. | A reusable prediction workflow needs an explicit input/evaluation contract; reproducing that example does not establish future-subject accuracy. See [workflow audit](condis-workflow-audit.md). |
| BCHM (#158) | File/report workflows and direct JAGS parity remain. | No additional advertised mathematical workflow was identified in this bounded review. See [BCHM guide](../docs/bchm.md). |
| DCT (#164) | Native report and rounding behavior remain unresolved. | Continuous/binary planning formulas are covered; preserve the documented discrepancy rather than add arbitrary participants. See [DCT guide](../docs/dct-normal.md). |
| PoP (#175) | HTML/Word/report parity remains. | Source-defined boundaries, selection and operating characteristics are implemented. See [PoP guide](../docs/pop-design.md). |
| STPLAN (#41) | Native automatic inverse bounds and integer allocation of proportional group totals are unspecified; session/report compatibility remains. | Current menu calculations and bounded inverse planning are implemented. An allocation convention would need to be explicit, not presented as recovered native behavior. See [planning guide](../docs/stplan-planning.md). |
| CID2BP (#38) | Native session/report interface and documented optimizer/certification limits remain. | All nine numerical options are implemented. See [CID2BP guide](../docs/cid2bp.md). |
| WFMM (#70) | Native prior/proposal defaults, `delta_omega` mapping, other transforms, compression and files remain unresolved. | The available fitting and prediction workflows do not establish these native contracts. See [WFMM audit](wfmm-audit.md) and [prediction guide](../docs/wfmm-prediction.md). |
| Pinnacle (#95) | Unsupported TIFF encodings, project ingestion, interactive editing and native reports remain. | These are input/application gaps; the reviewed peak-detection algorithm is covered. See [Pinnacle guide](../docs/pinnacle.md). |
| SurvivalContour (#166) | Recovered model families have Python counterparts; deployed app workflow and a missing stratified interval-contour uncertainty helper remain unverified. | Additional optional forest-library features are not automatically requirements of this app. See [source coverage boundary](survival-contour-coverage-audit.md). |

Dose Schedule Finder's guide also lists automatic calibration, synthetic
low-grade episode generation and within-patient adaptation policies as open;
the inspected contract does not specify complete procedures for those items.
See [its coverage limits](../docs/dose-schedule.md).

Next implementation choices should first identify a missing statistical
workflow with enough primary-source evidence to implement and verify it.
Reporting or input work can be valuable community functionality, but should
be named as such. Source uncertainties remain visible rather than being
silently replaced with arbitrary defaults or an assertion of native parity.
