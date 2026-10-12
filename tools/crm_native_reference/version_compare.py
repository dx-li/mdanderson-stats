"""Compare bounded original controller code after declared pointer relocation.

Does not establish equivalence of called inference routines or exception metadata.
Usage: python version_compare.py CRMKernel.dll BMA-CRMKernel.dll
"""

import hashlib
import json
import struct
import sys

import pefile

paths = sys.argv[1:]
assert len(paths) == 2
expected = (
    "8fc20c5f514d0c0a68877a55126ff97f4a43a88e3ded403adfe982b80bb501e7",
    "a170b501674307aee638b03915c5fb61337f7008a3a65c06b505098903e30768",
)
for path, digest in zip(paths, expected, strict=True):
    assert hashlib.sha256(open(path, "rb").read()).hexdigest() == digest
new, old = (pefile.PE(path) for path in paths)
# Explicit pointers observed in these regions; not arbitrary differing bytes.
pointers = {
    0x1003BD85: 0x1003BD55,
    0x100676CC: 0x100676EC,
    0x1000FB20: 0x1000FB10,
    0x1003300A: 0x10032FFA,
    0x1003BA4C: 0x1003BA1C,
    0x1003AC58: 0x1003AC28,
    0x1003AC90: 0x1003AC60,
    0x1003AD88: 0x1003AD58,
    0x1003ADD0: 0x1003ADA0,
}
rows = []
for name, start, end in (
    ("native_conductor", 0x316C0, 0x331F0),
    ("lookahead", 0x312D0, 0x316C0),
    ("dose_finder_region", 0x2A390, 0x2B100),
    ("cohort_helper", 0x331F0, 0x33300),
):
    a, b = new.get_data(start, end - start), old.get_data(start - 16, end - start)
    normalized = a
    for source, target in pointers.items():
        normalized = normalized.replace(struct.pack("<I", source), struct.pack("<I", target))
    assert normalized == b, name
    rows.append(
        {
            "region": name,
            "suite_rva": start,
            "older_rva": start - 16,
            "bytes": len(a),
            "suite_sha256": hashlib.sha256(a).hexdigest(),
            "older_sha256": hashlib.sha256(b).hexdigest(),
            "equal_after_declared_pointer_mapping": True,
        }
    )
print(json.dumps(rows, indent=2))
