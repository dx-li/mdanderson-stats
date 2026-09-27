# Next uncovered method: Dose Schedule Finder

Entry 75 is pending. The official page was verified during the UAROET
checkpoint: [Dose Schedule Finder 2.2.0](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/75),
last modified January 6, 2009. It explicitly identifies Braun, Thall, Nguyen
and de Lima (2007), *Simultaneously optimizing dose and schedule of a new
cytotoxic agent*, Clinical Trials 4:113–124, DOI 10.1177/1740774507076934.

The author hosts an accessible [published paper](https://odin.mdacc.tmc.edu/~pfthall/main/ClinTrials%20dose-sched%202007.pdf).
The web reader returned all 13 PDF pages. The official software uses a Bayesian
time-to-toxicity method to choose both per-administration dose and schedule.
The article permits actual administrations to differ from planned regimens.

Next: read the probability-model and allocation sections completely, verify
the per-administration hazard and prior parameterization, then delegate a
bounded probability/posterior/decision implementation to one Luna agent.
Inspect existing survival and dose-finding primitives before adding helpers.
Preserve serial numerical checks and explicit memory/work limits. Do not
substitute the distinct 2013 within-patient adaptation extension.

The native archive has not been retrieved or executed in this audit. No
equivalence or implemented-coverage claim is made at this stage.
