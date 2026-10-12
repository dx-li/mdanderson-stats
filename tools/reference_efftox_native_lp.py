"""Original current EffTox Lp shape bisection and utility instruction references.

Usage: python tools/reference_efftox_native_lp.py ORIGINAL_DLL > fixtures.json
The shared VM hash-pins the original DLL and supplies standard runtime math.
Only finite valid contour/probability inputs are covered; no exception parity.
"""

import json

from reference_efftox_native_contour import VM

rows = []
for e0, t0, e1, t1 in [(0.5, 0.65, 0.7, 0.25), (0.2, 0.7, 0.6, 0.35), (0.3, 0.6, 0.5, 0.1)]:
    vm = VM()
    # Complete original GetP, RootFinder constructor and bisection execute.
    shape = vm.call(129, [0, e0, t0, e1, t1])
    obj = vm.alloc()
    vm.mem[obj + 32] = 1
    vm.mem[obj + 40] = e0
    vm.mem[obj + 48] = t0
    vm.mem[obj + 56] = shape
    for efficacy, toxicity in [(e0, 0), (1, t0), (e1, t1), (1, 0), (0, 1), (0.9, 0.2), (0.2, 0.8)]:
        rows.append(
            {
                "contour": [e0, t0, e1, t1],
                "shape": shape,
                "efficacy": efficacy,
                "toxicity": toxicity,
                "utility": vm.call(128, [obj, efficacy, toxicity]),
            }
        )
print(json.dumps(rows, indent=2))
