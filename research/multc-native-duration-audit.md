# Multc Lean recovered native duration kernel

The official 2.1 archive was recovered on October 8, 2026. The MSI contains
the mixed native/managed x86 `MultcLeanCalcs.dll`. .NET method metadata locates
`RunDependentDurationSimulation` at RVA `0x75b0`. Its loop calls the single-trial
duration kernel at RVA `0x64d0`. Native outcome probabilities begin at offset
`0x28` of its scenario struct. Mean interarrival and window are at offsets
0 and 8. Managed property conversion and native category branches agree on
the order both, response only, toxicity only, neither.

## Evidence from original instructions

`tools/reference_multc_legacy_duration.py` checks the exact DLL SHA-256
`8ac45568e87530d4bb6732eb8807a22e4e8c5de52a5557a14f750c24867fa807`
and maps its PE image in Unicorn 2.1.4 x86-32. It runs original instructions
from RVA `0x64d0` through the normal return. Execution is bounded by 100,000
instructions and one second; incomplete execution is an error.

Only external services are substituted: uniform draws at RVA `0x14fa0`,
unit-mean exponential draws at `0x15030`, vector length/access at `0x10fd0` /
`0x17630`, and the unrelated security-cookie check at `0x25516`. The original
exponential-mean multiplication at RVA `0x15150` remains intact. So do its
window scaling, comparison/clipping, categories, counters, arrival loop,
balk logic, boundary tests and duration subtraction. Vector hooks validate
the requested object and index. No Python duration implementation is imported
by the reference generator, and no full Windows execution is claimed.

Thirty-four saved references record all supplied/consumed variates, counts and
duration. Counts and stream consumption match exactly. Python durations agree
within 2e-15 relative / 1e-14 absolute tolerance. The clipped response fixture
uses unit draws 20, 30 and 40: all follow-ups occur at the two-unit window,
with final duration 2.2. These are synthetic simulations, not patient data.

## Resolved contracts and limitations

The constant at native VA `0x1002dac8` is `2.99573227355399`, approximating
log(20). Responders draw an exponential with mean window/log(20), then clip
at the window. Nonresponders use the window. There is one follow-up time per
patient, with no separate toxicity-time draw. First enrollment is zero.

The compact response vector is indexed by responses. The toxicity vector is
indexed by **nontoxicities**. Before an outcome is generated, native flags
check whether its next sample size could stop using the previous latent
complete counts. If so, proposed arrivals before maximum follow-up balk and
consume additional exponential draws. Equality does not balk. After that,
the actual boundaries are checked; at the cap the routine skips this step.
Duration is maximum follow-up, rather than the later stop-evaluation time.

These details also show why the recovered simulation must be kept distinct
from observation-aware clinical conduct. It tracks generated latent outcomes
immediately, so pending responses can prevent a suspension before they are
observed. The Python compatibility API preserves this historical calculation
and exposes a patient ledger and separate decision time. It does not replace
the existing calendar engine's explicit and observation-aware policy.

The native wrapper screens zero-enrollment boundaries before calling this
kernel. The low-level explicit-boundary API requires callers to handle that
separately. The October 9 [boundary/aggregation continuation](multc-native-boundaries-audit.md)
now supplies an automatic design adapter, recovered cohort/minimum/prior
precedence and bounded legacy batch summaries. The subsequent
[managed-model/workflow audit](multc-lean-model-audit.md) completes saved-model,
default and protocol/report workflows and entry 12. Native integrator, CLR/GUI,
Windows RNG and original document-layout identity remain compatibility limits.

The original License.rtf prohibits redistributing the original program.
All original installers, DLLs and inspection runtimes stay in ignored local
research directories. Distribution contains independently written Python,
the authored reference harness and factual synthetic outputs only.

Reproduce with research-only `pefile==2024.8.26` and `unicorn==2.1.4` installed:

```bash
uv run --locked --extra plot python tools/reference_multc_legacy_duration.py \
  --dll research/raw/source-recovery-2026-10-08/multc-lean-native/MultcLeanCalcs.dll \
  --check
```
