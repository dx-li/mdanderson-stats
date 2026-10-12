# Native workflow recovery, 2026-10-12

Coverage remains 93 implemented / 39 partial / 6 pending. These recoveries close
specific algorithm gaps without declaring the whole applications complete.

## EffTox

See [fitter guide](../docs/efftox-native-contour.md). The current client DLL hash
`0947cfb848ded48fe5c147f0dd6a004acf3420f82791ab4208d613caa9ac99c2`
hides/disables the model group and unconditionally sets binary model 1 in
GetDataFrom_DoseFinderDialog (RVA0x9568). The prior calculator calls binary BBQ
calibration with six means/SDs; its model field is written but never read.
Shipped help contains no trinary workflow. Old XML retains trinary fields, so
this evidence applies to current GUI scope, not every historical version.

## BMA-CRM and CRM Suite

Both original desktop kernels show that pending-data DA safety flags cause
**wait**, preceding raw-rate caps. The Python `crm_suite` profile now implements
that behavior; paper and complete-data safety stops remain available.

Suite archive: https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/CRMSuite/CRM-Suite_v1.0.0.zip
SHA256 `f797abe0864dff1047f7ec8ef55745d2a9b19b155047e7cbdc951fa91916c872`;
kernel `8fc20c5f514d0c0a68877a55126ff97f4a43a88e3ded403adfe982b80bb501e7`.
BMA-CRM 2.2.4 kernel:
`a170b501674307aee638b03915c5fb61337f7008a3a65c06b505098903e30768`.
Suite safety branch RVA0x323ef–0x323fe; minimum wait 0x32400–0x32409;
raw gate 0x32eb3; old safety/minimum branches are 16 bytes earlier.
`tools/reference_crm_native_branches.py` regenerates the committed branch fixtures
from a separately obtained, hash-pinned Suite kernel. Original x86 probes under
Unicorn used supplied posterior flags,
matrix accessors and an explicit fixed RNG service. They establish controller
behavior, not posterior/MCMC/RNG parity. Raw cap ignores unobserved doses,
uses strict toxicity/observed > target, and scans intervening doses.

Native simulation begins at zero, batches fixed-dose cohorts, and retries a
wait after another exponential arrival gap. Current Python scheduling remains
explicitly different. The interactive conductor also checks within-cohort
safety, unlike simulation. Final native duration accounting and end-to-end
native simulations remain unverified; no scheduling shortcut was introduced.

## Adaptive Randomization

Official archive AdaptiveRandomization_V5.2.2.zip SHA256
`ff9c4d490a8876e1b3197d6fcb809bd8d001761be13f083a667e0e7db34fc877`;
ARand.Calcs DLL `a11d60157236bdf2e3395de39909dbb029b22bae52bcee5dd6d8ee4eebb3a6e3`.
Floor method RVA0x12e1a0, CheckTime RVA0xb4604, CheckMinMax RVA0x6f7c4,
CheckMinPrBest RVA0x715e0, StopTrial RVA0x70890, SimulateSingleTrial RVA0x59360.

Original x86 probability floor uses iterative proportional redistribution,
not additive mixture; 100 bisection cross-checks agreed within 3.34e-16.
Eight original instruction cases exercised all six monitoring modes with
substituted count/pending/evaluation services. Analyses occur at candidate
arrivals with overshoot, first arrival zero; duration completion is strict
current time > maximum and enrollment completion occurs at next candidate.
Final selection excludes suspended arms and breaks ties toward the later
map-ordered eligible arm. Full status/outcome/controller parity remains open.
The kernel uses the same configurable action for futility and loser suspension;
desktop defaults specify temporary closure. This conflicts with interpreting
guide futility as necessarily permanent and prevents an unsupported scope claim.

## ToxFinder

Official ToxFinder_V1.1.1.zip SHA256
`eddd41346542f3ee398a07eef8873a3687c3f3a9fa93e2a960ed217d01276d70`.
The self-extractor .WISE section contains an OLE/MSI signature at offset943;
extracting that embedded MSI recovered the application without installation.
The JAR is a GUI shell. Scientific core bin/ToxFinder.exe is stripped native
Compaq Visual Fortran. ExampleFiles.exe is an ordinary self-extracting ZIP with
281 files including cohort CSVs and TMML studies. Stage-two second-derivative
criterion remains unrecovered: do not regularize the published rank-one
information matrix or invent a determinant criterion.
