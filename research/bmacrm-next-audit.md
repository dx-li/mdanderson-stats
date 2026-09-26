# BMA-CRM next coverage audit (2026-09-26)

After the STPLAN survival batch, prioritize this pending scientific method over
inactive STPLAN interface features. Catalog entries 81 (BMA CRM), 132 (CRM Suite),
and 133 (online BMACRM) need separate version/capability checks; do not mark all
three covered from one posterior calculation.

Official current desktop page:
https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/81

The page identifies version 2.2.4 (2018-02-26), combining BMA-CRM and delayed-outcome
DA-CRM. Since version 2.1.2 skeleton inputs are **prior medians**, directly used
as power-model probabilities. The method PDF's earlier prior-mean elicitation
therefore does not describe the current default. Version 2.2.2 also adds a raw
fully-observed toxicity-rate escalation restriction and an MTD rule favoring
levels with at least three treated patients, with an explicit fallback if none
qualify. Read the full guide before implementing these decision rules.

Original method document:
https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/BMACRM/BMA-CRM_Description.pdf

Core model: for skeleton k, dose j, pi[k,j]=p[k,j]**exp(alpha[k]), with a normal
prior on alpha (documented mean zero, variance two). Binomial likelihoods give
one-dimensional posterior integrals and marginal likelihoods. Posterior model
weights combine prior model weights with these evidences; dose toxicity estimates
are their weighted posterior means. Safety stopping mixes the model-specific
posterior probabilities of lowest-dose toxicity exceeding the target. The PDF
also describes optional prior-mean-to-skeleton calibration and next-cohort
selection with no skipping of untried levels. These are scientific calculations,
not mere aliases to generic CRM routines.

Existing `bcrm_model.py` exposes log evidence but uses the Goodman CRM model and
a bounded uniform prior on beta. Inspect it for reusable numerical patterns, not
for assumed model equivalence. DA-CRM and simulated event timing remain separate
workflows requiring their own source audit. The newer online and CRM Suite
interfaces may differ from the older desktop package.
