"""Generate small references with the original Rice Wavelet Toolbox 2.4 C.

First run reference_pinnacle_filters.R. Place the fixed source archive recorded
in research/pinnacle-audit.md at research/raw/rwt-2.4.tar.gz. The archive/source
and compiled library remain reference-only, outside the package. The C shim
replaces MATLAB's allocation lifetime with tracked calloc/free; it changes no
transform arithmetic. Rice's conditions are in notices/rice-wavelet-LICENSE.txt.
"""

import csv
import ctypes
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "research/raw/rwt-2.4"
ARCHIVE_HASH = "cf242020f598718768f4fc6f7dfdcffee997caae1cf55b051af837afbf51b207"
data = (ROOT / "research/raw/rwt-2.4.tar.gz").read_bytes()
assert hashlib.sha256(data).hexdigest() == ARCHIVE_HASH
source_hashes = {}
with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
    for name in ("mrdwt_r.c", "mirdwt_r.c", "LICENSE"):
        stream = archive.extractfile("rwt/" + name)
        assert stream is not None
        content = stream.read()
        source_hashes[name] = hashlib.sha256(content).hexdigest()
        target = RAW / "rwt" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)

shim = r"""
#include <stdlib.h>
static void *allocations[32];
static int allocation_count = 0;
static void *reference_calloc(size_t n, size_t width) {
    void *p;
    if (allocation_count >= 32) abort();
    p = calloc(n, width);
    if (!p) abort();
    allocations[allocation_count++] = p;
    return p;
}
static void release_allocations(void) {
    while (allocation_count) free(allocations[--allocation_count]);
}
#define mxCalloc reference_calloc
int fpconv(double*,int,double*,double*,int,double*,double*);
int bpconv(double*,int,double*,double*,int,double*,double*);
#include "rwt/mrdwt_r.c"
#include "rwt/mirdwt_r.c"
void reference_forward(double *x, int m, int n, double *h, int k, int levels,
                       double *low, double *high) {
    MRDWT(x,m,n,h,k,levels,low,high);
    release_allocations();
}
void reference_inverse(double *x, int m, int n, double *h, int k, int levels,
                       double *low, double *high) {
    MIRDWT(x,m,n,h,k,levels,low,high);
    release_allocations();
}
"""
(RAW / "reference.c").write_text(shim)
library = RAW / "reference.dylib"
subprocess.run(
    [
        "clang",
        "-std=gnu89",
        "-Wno-return-type",
        "-Wno-implicit-int",
        "-O2",
        "-shared",
        "-fPIC",
        str(RAW / "reference.c"),
        "-o",
        str(library),
    ],
    check=True,
    timeout=30,
)
native = ctypes.CDLL(str(library))
pointer = ctypes.POINTER(ctypes.c_double)
for function in (native.reference_forward, native.reference_inverse):
    function.argtypes = [
        pointer,
        ctypes.c_int,
        ctypes.c_int,
        pointer,
        ctypes.c_int,
        ctypes.c_int,
        pointer,
        pointer,
    ]
    function.restype = None


def call(function, image, h, levels, low, high):
    function(
        image.ctypes.data_as(pointer),
        *image.shape,
        h.ctypes.data_as(pointer),
        len(h),
        levels,
        low.ctypes.data_as(pointer),
        high.ctypes.data_as(pointer),
    )


with (ROOT / "tests/fixtures/pinnacle-filters.csv").open(newline="") as stream:
    filters = {
        int(row["length"]): np.array(row["coefficients"].split("|"), dtype=float)
        for row in csv.DictReader(stream)
    }

cases = []
for name, shape, length, levels in (
    ("impulse", (8, 8), 8, 3),
    ("rectangular", (8, 16), 6, 3),
    ("haar", (8, 8), 2, 2),
):
    rr, cc = np.indices(shape)
    image = 2 + ((13 * rr + 7 * cc) % 17) / 3 + 30 * np.exp(-((rr - 3) ** 2 + (cc - 5) ** 2) / 4)
    if name == "impulse":
        image = np.full(shape, 2.0)
        image[3, 4] = 50
    image = np.asfortranarray(image)
    h = filters[length]
    low = np.empty(shape, order="F")
    high = np.empty((shape[0], shape[1] * 3 * levels), order="F")
    call(native.reference_forward, image, h, levels, low, high)
    reconstruction = np.empty(shape, order="F")
    call(native.reference_inverse, reconstruction, h, levels, low, high)
    np.testing.assert_allclose(reconstruction, image, rtol=1e-10, atol=1e-10)
    detail = np.array(
        [
            [high[:, (3 * level + b) * shape[1] : (3 * level + b + 1) * shape[1]] for b in range(3)]
            for level in range(levels)
        ]
    )
    outputs = {}
    for convention, divisor, multiplier in (("paper", 0.6745, 2), ("rwt", 0.67, 3.6)):
        center = np.median(detail[0, 2]) if convention == "paper" else 0.0
        sigma = float(np.median(np.abs(detail[0, 2] - center)) / divisor)
        threshold = multiplier * sigma
        keep = np.abs(high) >= threshold if convention == "paper" else np.abs(high) > threshold
        thresholded = np.asfortranarray(np.where(keep, high, 0))
        result = np.empty(shape, order="F")
        call(native.reference_inverse, result, h, levels, low, thresholded)
        outputs[convention] = {
            "sigma": sigma,
            "threshold": threshold,
            "multiplier": multiplier,
            "image": result.tolist(),
        }
    cases.append(
        {
            "case": name,
            "filter_length": length,
            "levels": levels,
            "input": image.tolist(),
            "low": low.tolist(),
            "detail": detail.tolist(),
            "denoised": outputs,
        }
    )

fixture = {
    "archive_sha256": ARCHIVE_HASH,
    "source_sha256": source_hashes,
    "reference": "Original RWT 2.4 forward/inverse C with allocation-only shim",
    "cases": cases,
}
(ROOT / "tests/fixtures/pinnacle-wavelet.json").write_text(json.dumps(fixture, indent=2) + "\n")
print(
    f"Generated {len(cases)} original-C transform references and {2 * len(cases)} denoising cases."
)
