"""Extract unchanged RF-SRC Brier helper C functions for tiny reference cases.

The harness compiles only the native KM/QE/gamma helpers, with a small array
adapter. It does not compile or invoke RF-SRC's tree-growing engine.
"""

from __future__ import annotations

import ctypes
import hashlib
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = Path("/Users/dxli2/math stats/mdanderson-stats/research/raw/randomForestSRC")
SOURCE = RAW / "src/randomForestSRC.c"
PINNED_SHA1 = "e9c6e896c4f93c6eb1b85bdf37992964d74376ea"


def _extract(source: str, name: str) -> str:
    match = re.search(rf"^void\s+{name}\s*\(", source, re.M)
    if match is None:
        raise ValueError(f"missing source helper {name}")
    start = source.index("{", match.start())
    depth = 1
    end = start + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[match.start() : end] + "\n"


def _native_library() -> ctypes.CDLL:
    raw = SOURCE.read_bytes()
    blob = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
    if blob != PINNED_SHA1:
        raise ValueError("RF-SRC C source does not match the pinned helper reference")
    source = raw.decode()
    header = r"""
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdlib.h>
typedef unsigned int uint;
typedef struct { uint nodeID; } Node;
enum alloc_type { NRUTIL_DPTR = 1 };
#define SURV_BSG1 20
#define TRUE 1
#define FALSE 0
#define RF_nativeNaN NAN
#define RF_nativeIsNaN(x) isnan(x)
static uint RF_splitRule, RF_quantileSize;
static uint RF_masterTimeSize;
static double *RF_quantile;
static uint **RF_masterTimeIndex;
static double **RF_status;
static uint *RF_masterIndexStorage[2];
static double *RF_statusStorage[2];
static void RF_nativeError(const char *format, ...) { (void)format; abort(); }
static void RF_nativeExit(void) { abort(); }
static double *dvector(long lo, long hi) {
  (void)lo; return (double*)calloc((size_t)hi + 1, sizeof(double));
}
static void free_dvector(double *p, long lo, long hi) {
  (void)lo; (void)hi; free(p);
}
static uint *uivector(long lo, long hi) {
  (void)lo; return (uint*)calloc((size_t)hi + 1, sizeof(uint));
}
static void free_uivector(uint *p, long lo, long hi) {
  (void)lo; (void)hi; free(p);
}
static void **new_vvector(unsigned long long lo, unsigned long long hi,
                          enum alloc_type type) {
  (void)lo; (void)type; return (void**)calloc((size_t)hi + 1, sizeof(void*));
}
static void free_new_vvector(void *p, unsigned long long lo,
                             unsigned long long hi, enum alloc_type type) {
  (void)lo; (void)hi; (void)type; free(p);
}
"""
    functions = "".join(
        _extract(source, name)
        for name in (
            "stackAndGetSplitSurv2",
            "unstackAndGetSplitSurv2",
            "stackAndGetQETime",
            "unstackQETime",
            "stackAndGetLocalGamma",
            "unstackLocalGamma",
        )
    )
    wrapper = r"""
void reference_brier(uint n, double *time, double *event, double prob,
                     uint *qe_out, double *gamma_out) {
  uint *master=calloc(n+1,sizeof(uint));
  uint *event_idx=calloc(n+1,sizeof(uint));
  uint *event_count=calloc(n+1,sizeof(uint));
  uint *event_risk=calloc(n+1,sizeof(uint));
  uint *censor_idx=calloc(n+1,sizeof(uint));
  uint *censor_count=calloc(n+1,sizeof(uint));
  uint *censor_risk=calloc(n+1,sizeof(uint));
  uint m=0, cm=0, unique=0, master_size=0;
  double *event_surv=NULL, *censor_surv=NULL;
  double *prob_storage=calloc(2,sizeof(double));
  uint *qe=NULL, qe_size=0;
  double *gHat=NULL, **weights=NULL, **gamma=NULL;
  uint *rows=calloc(n+1,sizeof(uint));
  uint *nonmiss=calloc(n+1,sizeof(uint));
  Node node={0};
  for(uint i=1;i<=n;i++) {
    unique=0;
    for(uint j=1;j<i;j++) if(time[j]==time[i]) { unique=master[j]; break; }
    if(!unique) {
      unique=1;
      for(uint j=1;j<i;j++) if(time[j]<time[i]) unique++;
    }
    master[i]=unique;
    if(unique>master_size) master_size=unique;
  }
  RF_masterTimeSize=master_size;
  for(uint level=1;level<=master_size;level++) {
    for(uint i=1;i<=n;i++) if(master[i]==level) {
      if(event[i]>0) {
        if(!m || event_idx[m]!=level) event_idx[++m]=level;
        event_count[m]++;
      } else {
        if(!cm || censor_idx[cm]!=level) censor_idx[++cm]=level;
        censor_count[cm]++;
      }
    }
  }
  for(uint j=1;j<=m;j++) for(uint i=1;i<=n;i++) event_risk[j]+=(master[i]>=event_idx[j]);
  for(uint j=1;j<=cm;j++) for(uint i=1;i<=n;i++) censor_risk[j]+=(master[i]>=censor_idx[j]);
  RF_splitRule=SURV_BSG1; RF_quantileSize=1;
  prob_storage[1]=prob; RF_quantile=prob_storage;
  RF_masterIndexStorage[1]=master; RF_masterTimeIndex=RF_masterIndexStorage;
  RF_statusStorage[1]=event; RF_status=RF_statusStorage;
  for(uint i=1;i<=n;i++) rows[i]=nonmiss[i]=i;
  qe_out[0]=0;
  for(uint i=1;i<=n;i++) gamma_out[i-1]=0.0;
  if(m>0) {
    stackAndGetSplitSurv2(1,&node,m,event_count,event_risk,&event_surv);
    stackAndGetSplitSurv2(1,&node,cm,censor_count,censor_risk,&censor_surv);
    stackAndGetQETime(1,&node,event_idx,m,event_surv,&qe,&qe_size);
    qe_out[0]=qe[1];
    stackAndGetLocalGamma(1,&node,rows,n,nonmiss,n,event_idx,m,
                          censor_idx,cm,censor_surv,qe,qe_size,
                          &gHat,&weights,&gamma);
    if(qe[1]>0 && gamma[qe[1]]!=NULL)
      for(uint i=1;i<=n;i++) gamma_out[i-1]=gamma[qe[1]][i];
    unstackLocalGamma(1,n,event_idx,m,qe,qe_size,gamma);
    unstackQETime(1,m,qe);
    unstackAndGetSplitSurv2(1,&node,m,event_surv);
    unstackAndGetSplitSurv2(1,&node,cm,censor_surv);
  }
  free(master); free(event_idx); free(event_count); free(event_risk);
  free(censor_idx); free(censor_count); free(censor_risk); free(rows);
  free(nonmiss); free(prob_storage);
}
"""
    generated = RAW / "reference-brier-kernels.c"
    generated.write_text(header + functions + wrapper)
    library = RAW / "reference-brier-kernels.so"
    kind = "-dynamiclib" if sys.platform == "darwin" else "-shared"
    subprocess.run(
        ["cc", "-O0", "-fPIC", kind, str(generated), "-o", str(library), "-lm"],
        check=True,
        timeout=60,
    )
    result = ctypes.CDLL(str(library))
    pointer = ctypes.POINTER(ctypes.c_double)
    result.reference_brier.argtypes = [
        ctypes.c_uint,
        pointer,
        pointer,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_uint),
        pointer,
    ]
    result.reference_brier.restype = None
    return result


def reference_case(
    library: ctypes.CDLL,
    time: list[float],
    event: list[float],
    prob: float,
) -> tuple[int, list[float]]:
    n = len(time)

    def values(items: list[float]) -> ctypes.Array[ctypes.c_double]:
        return (ctypes.c_double * (n + 1))(0.0, *items)

    qe = ctypes.c_uint()
    gamma = (ctypes.c_double * n)()
    library.reference_brier(n, values(time), values(event), prob, ctypes.byref(qe), gamma)
    return int(qe.value), list(gamma)


def main() -> None:
    import csv

    cases = {
        "shared_failure_weight": ([1, 1.5, 2, 3], [1, 0, 1, 0], [0, 0, 1, 1], 0.9, 0.5),
        "threshold_equality_zero": ([1, 2], [1, 0], [0, 1], 0.5, 0.5),
        "first_event_after_censor": ([1, 2, 3], [0, 1, 0], [0, 1, 1], 0.9, 0.5),
        "equality_uses_previous_point": ([1, 2, 3, 4], [1, 1, 0, 0], [0, 0, 1, 1], 0.5, 0.5),
        "censor_tied_with_event": ([1, 1, 2, 3], [1, 0, 1, 0], [0, 0, 1, 1], 0.9, 0.5),
    }
    library = _native_library()
    output = ROOT / "tests/fixtures/random-survival-brier-native.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["case", "row", "time", "event", "left", "prob", "qe_index", "gamma", "score"]
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name, (time, event, x, prob, cut) in cases.items():
            qe, gamma = reference_case(library, time, event, prob)
            left = [value <= cut for value in x]
            if qe == 0:
                score = 0.0
            else:
                n_left = sum(left)
                score = (n_left / len(left)) * (
                    sum(value for value, keep in zip(gamma, left, strict=True) if keep) / n_left
                ) ** 2
                n_right = len(left) - n_left
                score += (n_right / len(left)) * (
                    sum(value for value, keep in zip(gamma, left, strict=True) if not keep)
                    / n_right
                ) ** 2
            observations = zip(time, event, gamma, strict=True)
            for index, (duration, status, value) in enumerate(observations, 1):
                writer.writerow(
                    {
                        "case": name,
                        "row": index,
                        "time": duration,
                        "event": status,
                        "left": left[index - 1],
                        "prob": prob,
                        "qe_index": qe,
                        "gamma": value,
                        "score": score,
                    }
                )
    print(output)


if __name__ == "__main__":
    main()
