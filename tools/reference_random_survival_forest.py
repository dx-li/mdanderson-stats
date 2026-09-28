"""Run unchanged randomForestSRC kernels without installing the R package.

Sources stay in ignored research/raw/randomForestSRC. This runs split/leaf
kernels, not the full native tree-growing or forest engine.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import re
import resource
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "research/raw/randomForestSRC"
PIN = "b4d099e262423362a8872c13c468e6dbe2f9e9da"
HASHES = {
    "src/randomForestSRC.c": "e9c6e896c4f93c6eb1b85bdf37992964d74376ea",
    "src/splitCustom.c": "337a084d630f4871a61efd37caaa67ae1cfb2e6b",
    "R/utilities.survival.R": "9ed62c12c118edd11d78fb85f2ec230448646406",
    "src/randomForestSRC.h": "6355ed999d06447747f78b01ee2a370e6bc43d1e",
}


def native_function(source: str, name: str) -> str:
    match = re.search(rf"^(?:void|double) {name}\s*\(", source, re.M)
    if match is None:
        raise ValueError(f"missing native function {name}")
    start = source.index("{", match.start())
    depth = 1
    end = start + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[match.start() : end] + "\n"


def library() -> ctypes.CDLL:
    sources = {}
    for name, expected in HASHES.items():
        value = (RAW / name).read_bytes()
        digest = hashlib.sha1(f"blob {len(value)}\0".encode() + value).hexdigest()
        if digest != expected:
            raise ValueError(f"source hash mismatch: {name}")
        sources[name] = value.decode()
    header = r"""
#include <math.h>
#include <stdlib.h>
typedef unsigned int uint;
#define LEFT 1
#define TRUE 1
#define FALSE 0
#define OPT_COMP_RISK 1
#define EPSILON 1.0e-9
#define RF_nativeNaN NAN
typedef struct {
  uint membrCount, eTimeSize, nodeID;
  uint *eventTimeIndex, *atRiskCount, **eventCount;
  double **localRatio, *localSurvival, *localNelsonAalen;
} Terminal;
static uint RF_eventTypeSize=1, RF_opt=0, RF_sortedTimeInterestSize=0;
static double *RF_masterTime, *RF_timeInterest;
static void RF_nativeError(const char *msg, ...) { abort(); }
static void RF_nativeExit(void) { abort(); }
static uint *alloc_uivector(uint n) { return calloc(n+1,sizeof(uint)); }
static void dealloc_uivector(uint *p, uint n) { free(p); }
static void stackLocalRatio(Terminal *p, uint types, uint n) {
  p->localRatio=calloc(types+1,sizeof(double*));
  for(uint j=1;j<=types;j++) p->localRatio[j]=calloc(n+1,sizeof(double));
}
static void stackLocalSurvival(Terminal *p, uint n) {
  p->localSurvival=calloc(n+1,sizeof(double));
}
static void stackLocalNelsonAalen(Terminal *p, uint n) {
  p->localNelsonAalen=calloc(n+1,sizeof(double));
}
"""
    extracted = native_function(sources["src/splitCustom.c"], "getCustomSplitStatisticSurvival")
    for name in (
        "getLocalRatio",
        "getLocalSurvival",
        "getLocalNelsonAalen",
        "mapLocalToTimeInterest",
        "getConcordanceIndex",
    ):
        extracted += native_function(sources["src/randomForestSRC.c"], name)
    wrapper = r"""
/* Harness arrays are one-based to preserve the original kernel bodies. */
double reference_split(uint n, double *t, double *e, char *left) {
  double *et=calloc(n+1,sizeof(double)); uint m=0;
  for(uint i=1;i<=n;i++) if(e[i]>0 && (!m || t[i]!=et[m])) et[++m]=t[i];
  double value=getCustomSplitStatisticSurvival(n,left,t,e,1,m,et,NULL,0,0,0,NULL,0);
  free(et); return value;
}
void reference_curve(uint n, double *t, double *e, uint q, double *grid,
                     double *s, double *h) {
  Terminal p={0}; p.membrCount=n;
  RF_masterTime=calloc(n+1,sizeof(double));
  for(uint i=1;i<=n;i++) if(e[i]>0 && (!p.eTimeSize || t[i]!=RF_masterTime[p.eTimeSize]))
    RF_masterTime[++p.eTimeSize]=t[i];
  p.eventTimeIndex=alloc_uivector(p.eTimeSize);
  p.atRiskCount=alloc_uivector(p.eTimeSize);
  p.eventCount=calloc(2,sizeof(uint*));
  p.eventCount[1]=alloc_uivector(p.eTimeSize);
  for(uint j=1;j<=p.eTimeSize;j++) {
    p.eventTimeIndex[j]=j;
    for(uint i=1;i<=n;i++) {
      p.atRiskCount[j]+=(t[i]>=RF_masterTime[j]);
      p.eventCount[1][j]+=(e[i]>0 && t[i]==RF_masterTime[j]);
    }
  }
  getLocalRatio(1,&p); getLocalSurvival(1,&p); getLocalNelsonAalen(1,&p);
  RF_sortedTimeInterestSize=q; RF_timeInterest=grid;
  for(uint j=1;j<=q;j++) { s[j]=1; h[j]=0; }
  mapLocalToTimeInterest(1,&p,p.localSurvival,s);
  mapLocalToTimeInterest(1,&p,p.localNelsonAalen,h);
  if(p.localRatio) { free(p.localRatio[1]); free(p.localRatio); }
  free(p.localSurvival); free(p.localNelsonAalen); free(p.eventTimeIndex);
  free(p.atRiskCount); free(p.eventCount[1]); free(p.eventCount); free(RF_masterTime);
}
"""
    target = RAW / "reference-kernels.c"
    target.write_text(header + extracted + wrapper)
    shared = RAW / "reference-kernels.so"
    kind = "-dynamiclib" if sys.platform == "darwin" else "-shared"
    subprocess.run(
        ["cc", "-O0", "-fPIC", kind, str(target), "-o", str(shared), "-lm"],
        check=True,
        timeout=60,
    )
    lib = ctypes.CDLL(str(shared))
    pointer = ctypes.POINTER(ctypes.c_double)
    lib.reference_split.argtypes = [ctypes.c_uint, pointer, pointer, ctypes.POINTER(ctypes.c_char)]
    lib.reference_split.restype = ctypes.c_double
    lib.reference_curve.argtypes = [
        ctypes.c_uint,
        pointer,
        pointer,
        ctypes.c_uint,
        pointer,
        pointer,
        pointer,
    ]
    lib.reference_curve.restype = None
    lib.getConcordanceIndex.argtypes = [
        ctypes.c_int,
        ctypes.c_uint,
        pointer,
        pointer,
        pointer,
        pointer,
    ]
    lib.getConcordanceIndex.restype = ctypes.c_double
    return lib


def doubles(values: list[float]) -> ctypes.Array:
    return (ctypes.c_double * (len(values) + 1))(0, *values)


def reference_tree(lib: ctypes.CDLL, case: dict, nodesize: int) -> dict:
    """Independent exhaustive one-feature tree driver using native C scores.

    No sampling or feature randomness is involved. This intentionally does not
    call the Python package or claim execution of the native tree driver.
    """
    nodes = []
    times, events, x, grid = (case[key] for key in ("time", "event", "x", "times"))

    def grow(rows: list[int]) -> int:
        index = len(nodes)
        node = {"size": len(rows), "cut": None, "left": None, "right": None}
        nodes.append(node)
        t = doubles([times[i] for i in rows])
        e = doubles([events[i] for i in rows])
        observed_events = [events[i] for i in rows]
        stop = (
            len(rows) < 2 * nodesize
            or not any(observed_events)
            or (all(observed_events) and len({times[i] for i in rows}) == 1)
        )
        best, cut = None, None
        if not stop:
            for candidate in sorted({x[i] for i in rows})[:-1]:
                membership = [1 if x[i] <= candidate else 2 for i in rows]
                left = (ctypes.c_char * (len(rows) + 1))(0, *membership)
                score = lib.reference_split(len(rows), t, e, left)
                if best is None or score - best > 1e-9:
                    best, cut = score, candidate
        if cut is None:
            s, h = doubles([0] * len(grid)), doubles([0] * len(grid))
            lib.reference_curve(len(rows), t, e, len(grid), doubles(grid), s, h)
            node.update(survival=list(s)[1:], cumulative_hazard=list(h)[1:])
        else:
            node["cut"] = cut
            node["left"] = grow([i for i in rows if x[i] <= cut])
            node["right"] = grow([i for i in rows if x[i] > cut])
        return index

    grow(list(range(len(times))))
    profiles = [-2, -0.8, 0, 0.5, 1.8, 4, 10]
    survival, hazard = [], []
    for value in profiles:
        node = nodes[0]
        while node["cut"] is not None:
            node = nodes[node["left"] if value <= node["cut"] else node["right"]]
        survival.append(node["survival"])
        hazard.append(node["cumulative_hazard"])
    return {
        "case": case["name"],
        "nodesize": nodesize,
        "nodes": nodes,
        "profiles": profiles,
        "survival": survival,
        "cumulative_hazard": hazard,
    }


def main() -> None:
    started = time.perf_counter()
    lib = library()
    grid = [0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 20]
    cases = [
        (
            "tied",
            [1, 1, 2, 2, 2, 3, 4, 4, 5, 6, 7, 8],
            [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0],
            [0.2, -1, 0.5, 2, -0.8, 1.5, 0, 1, 3, -0.5, 2.5, 4],
        ),
        ("all_events", [1, 1, 2, 2, 3, 4, 5, 6], [1] * 8, [0, 1, 2, 3, 4, 5, 6, 7]),
        ("all_censored", [1, 2, 2, 3, 4, 5], [0] * 6, [0, 1, 2, 3, 4, 5]),
        ("one_event", [1, 2, 2, 3, 4, 5], [0, 1, 0, 0, 0, 0], [0, 1, 2, 3, 4, 5]),
        ("same_time", [2] * 6, [1, 0, 1, 0, 1, 0], [0, 1, 2, 3, 4, 5]),
        ("zero_time", [0, 0, 1, 2, 2, 3], [1, 0, 1, 1, 0, 0], [0, 1, 2, 3, 4, 5]),
        (
            "duplicates",
            [1, 1, 1, 2, 2, 2, 3, 3, 4, 5],
            [1, 1, 1, 0, 0, 1, 1, 1, 0, 1],
            [0, 0, 0, 1, 1, 2, 3, 3, 4, 5],
        ),
    ]
    output = []
    for name, times, events, x in cases:
        t, e = doubles(times), doubles(events)
        s, h = doubles([0] * len(grid)), doubles([0] * len(grid))
        lib.reference_curve(len(times), t, e, len(grid), doubles(grid), s, h)
        splits = []
        for cut in sorted(set(x))[:-1]:
            membership = [1 if value <= cut else 2 for value in x]
            left = (ctypes.c_char * (len(times) + 1))(0, *membership)
            score = lib.reference_split(len(times), t, e, left)
            splits.append({"cut": cut, "score": score})
        output.append(
            {
                "name": name,
                "time": times,
                "event": events,
                "x": x,
                "times": grid,
                "survival": list(s)[1:],
                "cumulative_hazard": list(h)[1:],
                "splits": splits,
            }
        )
    r_source = RAW / "R/utilities.survival.R"
    r_code = (
        f"source({json.dumps(str(r_source))}); "
        "y<-cbind(1:401,rep(1,401)); "
        "for(n in c(0,1,5,150)) { "
        "g<-get.grow.event.info(y,'surv',ntime=n)$time.interest; "
        "cat(n, paste(g,collapse=','),sep=';'); cat('\\n') }"
    )
    r_output = subprocess.run(
        ["Rscript", "--vanilla", "-e", r_code],
        check=True,
        timeout=30,
        capture_output=True,
        text=True,
    ).stdout
    time_grids = []
    for line in r_output.splitlines():
        ntime, values = line.split(";")
        time_grids.append(
            {
                "ntime": int(ntime),
                "event_times": list(range(1, 402)),
                "expected": [float(value) for value in values.split(",")],
            }
        )
    trees = [
        reference_tree(lib, case, nodesize)
        for case in output
        if case["name"] in ("tied", "duplicates")
        for nodesize in (1, 2, 15)
    ]
    target = ROOT / "tests/fixtures/random-survival-forest-native.json"
    target.write_text(
        json.dumps(
            {
                "source_commit": PIN,
                "source_blobs": HASHES,
                "cases": output,
                "time_grids": time_grids,
                "trees": trees,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    usage = resource.getrusage(resource.RUSAGE_SELF)
    child = resource.getrusage(resource.RUSAGE_CHILDREN)
    divisor = 1024**2 if sys.platform == "darwin" else 1024
    print(
        {
            "cases": len(output),
            "split_scores": sum(len(case["splits"]) for case in output),
            "deterministic_tree_cases": len(trees),
            "seconds": time.perf_counter() - started,
            "peak_MiB": usage.ru_maxrss / divisor,
            "child_peak_MiB": child.ru_maxrss / divisor,
            "swaps": usage.ru_nswap + child.ru_nswap,
        }
    )


if __name__ == "__main__":
    main()
