# Bounded original CRM conduct references

These research-only harnesses require the original locally extracted libraries;
no DLL or translated native instruction body is distributed. `patient_reference.py`
hash-pins CRM Suite 1.0.0 and BMA-CRM 2.2.4 before executing original IL.

Run `python verify.py /path/CRMKernel.dll /path/BMA-CRMKernel.dll` in an environment
with dnfile, dncil and Unicorn. It compares all 142 generated cases per version
byte-for-byte with `research/fixtures/crm-native`.

- 60 patient outcome/censor cases execute `CalcOutcomeAndCensoringTime` at RVA
  0xe228 (Suite) / 0xe214 (older). Runtime vector/pair/sort operations are supplied.
  Dates are integral days, so the native 1e-6 comparator tie tolerance reduces
  to exact ties. Partitioned hazard exposure is skipped **after** outcome and
  censor writes; these fixtures do not validate that exposure routine.
- 32 cohort-size cases execute the complete x86 helper at 0x331f0 (Suite),
  with older addresses 16 bytes earlier. Two additional cases execute only the
  DA within-cohort safety disposition slice 0x3213b..0x3214a.
- 48 managed replay-gate cases execute IL346..588 of `GetDecision` at RVA
  0xe634 / 0xe620, supplying patient/model getters and omitting descriptive-string
  setup. They establish the gate and preceding-patient prefix, not the posterior.
- `ManagedPatientProbe.cs` uses the original Utilities Patient class through
  reflection and XmlSerializer. Compile with a C# compiler, then run with
  Utilities.dll and an output directory as its two arguments. It generates and
  reopens the en-US and en-GB synthetic patient XML. Both original versions give
  the four preserved XML files byte-for-byte under isolated Mono.

The outcome method bodies are identical across these versions (1036 bytes,
SHA256 9a4655a4712eafce5201f147df799b9b79bc5b07f2dfc84bc8dc8f14007129e2).
The managed GetDecision bodies are also identical (1960 bytes,
SHA256 1c66c153562e4be52592dd240f61f0ef4559be4606c94c840b1707b0d60a9b26).
This is bounded evidence, not an assertion that both entire kernels are identical.

Source archives, fetched from the official catalog:

- https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/CRMSuite/CRM-Suite_v1.0.0.zip
  SHA256 f797abe0864dff1047f7ec8ef55745d2a9b19b155047e7cbdc951fa91916c872
  Kernel SHA256 8fc20c5f514d0c0a68877a55126ff97f4a43a88e3ded403adfe982b80bb501e7
  Utilities SHA256 05cfd83a07d997af320cb2ecb5f8a1c58a23578484267b5b130adc8c21d945c9
- https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/BMACRM/BMA-CRMSimulator_V2.2.4.zip
  SHA256 1f8a00c8766bdcfbb569ec4990c29c91b026ab54215a6b1436bc90b6361a79c3
  Kernel SHA256 a170b501674307aee638b03915c5fb61337f7008a3a65c06b505098903e30768
  Utilities SHA256 5075d07fa4a2aeea4afd9dc339267a56f6777c8c33501427aa830c4e52deb683

Original terms remain applicable. These probes do not establish clinical validity,
full application equivalence, or matching MCMC/random streams.

`version_compare.py` checks four bounded native regions across the two versions
with an explicit mapping of embedded pointers. The conductor, lookahead and
dose-finder regions then match; the cohort helper matches before mapping.
`version-regions.json` records hashes and lengths. This comparison establishes
local control/arithmetic identity, not equivalence of called inference runtimes
or exception metadata.
