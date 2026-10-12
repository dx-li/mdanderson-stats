# PRT1.0 native recovery

Current status69 remains partial. Original assets are locally recovered under
ignored `research/raw/next-desktop-2026-10-12/PRT_V1.0/PRT` and are not redistributed.
Conduct SHA256: `2943f71ab27e300c1168b2d9f6a93815b75efe753b6af2b543c1370d077a1123`.
Simulation SHA256: `52903a7af2e75998bf2a8cbda6e2049e7fa25e0981cf9806317841da13a91b54`.

## Projection contract recovered

`tools/reference_prt_projection.py` executes conduct RVA36fb–434c, including
actual native matrix inversion961d, on supplied synthetic posterior risk draws.
Thirty-two cases cover2/3/4/6/10doses, independent and strongly correlated draws,
negative weighted averages and weighted averages above one. These are standalone
projection references, not posterior, RNG or full trial parity.

Native behavior is **maximum over lower block endpoints of minimum over upper
block endpoints**, followed by a lower floor at zero (4267–4296). The final
published paper uses the reverse min/max ordering in Section3, Step3:
[author-hosted paper](https://odin.mdacc.tmc.edu/~pfthall/main/Biostatistics_LateTox_2008.pdf).
The current Python paper API correctly retains that published ordering. With
negative inverse-covariance weights the two orderings need not agree: one
synthetic10-dose case differs by0.04992356. Do not silently substitute the
native ordering into the paper API.

The lower floor does not establish probability validity. Four native cases still
return values above1.005 (maximum1.01016399). The paper API currently rejects the
unbounded transformation in these cases, which is safer than feeding invalid
probabilities to predictive calculations. Native compatibility must be an explicit
profile with diagnostics; a mathematically valid alternative would require its
own declared contract and validation, not an unlabelled clipping change.

Independent NumPy covariance/block solves reproduce the native max-min/lower-floor
outputs to4e-13 on this grid. The fixture is portable synthetic data; no original
executable content is included.

## Simulation generator recovery in progress

The simulation executable's b786 helper evaluates a beta density in log space
using alpha/beta and supplied interior x. Its calibration caller at24f8–274d
evaluates100 locations stepped by1/101, divides by a searched scale, and matches
`1-exp(-integrated hazard)` to the configured toxicity probability within a
native single-precision1e-5 tolerance. Exact time-grid endpoints and subsequent
inverse-survival generation still need executable fixtures before implementation.
Do not substitute a simple beta event-time distribution: the observed code uses
beta-shaped hazard rates. Scratch disassembly/probes live under`/tmp/prt-*`.

Remaining scope: explicit projection compatibility/validity policy; exact native
patient-input conversion; calibrated hazard/event-time generator and accrual/
suspension timing; full saved study, conduct/simulation/report workflows and
independent acceptance. No application promotion from these source findings.
