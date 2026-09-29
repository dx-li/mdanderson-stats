"""Regenerate tiny categorical/anti-routing references from pinned native C.

Requires the ignored randomForestSRC source cache described in the audit.
Extracted native sources and compiled objects stay under research/raw.
"""

from __future__ import annotations

import hashlib
import json
import re
import resource
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "research/raw/randomForestSRC/src/randomForestSRC.c"
HEADER = ROOT / "research/raw/randomForestSRC/src/randomForestSRC.h"
EXPECTED_GIT_BLOB = "e9c6e896c4f93c6eb1b85bdf37992964d74376ea"
EXPECTED_HEADER_BLOB = "6355ed999d06447747f78b01ee2a370e6bc43d1e"
OUT = ROOT / "research/raw/random-survival-extensions/reference.c"
FIXTURE = ROOT / "tests/fixtures/random-survival-extensions-native.csv"


def native_function(source: str, name: str) -> str:
    match = re.search(rf"^(?:void|char|Node \*)\s*{name}\s*\(", source, re.M)
    if match is None:
        raise ValueError(f"missing native function {name}")
    start = source.index("{", match.start())
    depth, end = 1, start + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[match.start() : end] + "\n"


def main() -> None:
    start = time.monotonic()
    data = SOURCE.read_bytes()
    actual = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    if actual != EXPECTED_GIT_BLOB:
        raise ValueError(f"source pin mismatch: expected {EXPECTED_GIT_BLOB}, got {actual}")
    header = HEADER.read_bytes()
    header_actual = hashlib.sha1(f"blob {len(header)}\0".encode() + header).hexdigest()
    if header_actual != EXPECTED_HEADER_BLOB:
        raise ValueError(
            f"header pin mismatch: expected {EXPECTED_HEADER_BLOB}, got {header_actual}"
        )
    source = data.decode()
    prefix = r"""
#include <stdio.h>
#include <stdlib.h>
typedef unsigned int uint;
#define LEFT 1
#define RIGHT 2
#define TRUE 1
#define FALSE 0
#define MAX_EXACT_LEVEL sizeof(uint) * 8
#define RF_nativeIsNaN(x) 0
struct splitInfo;
typedef struct splitInfo SplitInfo;
typedef struct node Node;
typedef struct factor Factor;
struct splitInfo { int *randomVar; uint *mwcpSizeAbs; void **randomPts; uint hcDim; };
struct node { Node *left, *right; SplitInfo *splitInfo; uint repMembrSize, id; };
struct factor { uint r, cardinalGroupCount; uint *cardinalGroupSize; uint **cardinalGroupBinary; };
static double RF_vimpThreshold;
static char RF_importanceFlag[8];
static double uniform_tape[16];
static uint uniform_n, uniform_at;
static double ran1D(uint treeID) { (void)treeID; return uniform_tape[uniform_at++]; }
static uint upower(uint base, uint exp) { uint out=1; while(exp--) out*=base; return out; }
static uint ulog2(uint x) { uint n=0; while(x>1) { x>>=1; ++n; } return n; }

static char getDaughterPolarity(uint treeID, SplitInfo *info, uint individual, void *value) {
  double **x = (double **)value;
  double *cut = (double *)info->randomPts[1];
  (void)treeID;
  return ((cut[1] - x[info->randomVar[1]][individual]) >= 0.0) ? LEFT : RIGHT;
}
"""
    functions = "\n".join(
        native_function(source, name)
        for name in ("bookPair", "splitOnFactor", "antiMembershipGeneric")
    )
    suffix = r"""
static uint choose(uint n, uint k) {
  if (k > n-k) k=n-k;
  uint v=1;
  for(uint i=1;i<=k;i++) v=v*(n-k+i)/i;
  return v;
}
static void run_factor(uint r) {
  uint groups=r/2, *sizes=calloc(groups+1,sizeof(uint)), **binary=calloc(groups+1,sizeof(uint*));
  uint *levels=calloc(r+1,sizeof(uint));
  for(uint g=1;g<=groups;g++) {
    sizes[g]=choose(r,g);
    if(!(r&1) && g==groups) sizes[g]/=2;
    binary[g]=calloc(sizes[g]+1,sizeof(uint));
  }
  Factor f={r,groups,sizes,binary};
  for(uint g=1;g<=groups;g++) {
    uint row=0;
    for(uint j=0;j<=r;j++) levels[j]=0;
    bookPair(r,g,1,&row,levels,&f);
    for(uint j=1;j<=row;j++) printf("enum,%u,%u,%u,0\n",r,g,binary[g][j]);
  }
  for(uint g=1;g<=groups;g++) free(binary[g]);
  free(binary); free(sizes); free(levels);
}
int main(void) {
  puts("kind,a,b,c,d");
  run_factor(3); run_factor(4);
  /* Anti-routing uses one target split; caller supplies explicit uniforms. */
  double values[3] = {0.0, 0.25, 0.75};
  double cut[2] = {0.0, 0.5};
  double *x[2] = {NULL, values};
  int vars[2] = {0, 1};
  uint msize[2] = {0, 0};
  void *points[2] = {NULL, cut};
  SplitInfo si = {vars, msize, points, 0};
  Node left = {NULL, NULL, NULL, 0, 1}, right = {NULL, NULL, NULL, 0, 2};
  Node root = {&left, &right, &si, 2, 0};
  RF_importanceFlag[1] = TRUE;
  for (uint qcase=0; qcase<3; ++qcase) {
    double q = qcase == 0 ? 1.0 : 0.5;
    double u = qcase == 0 ? 0.25 : (qcase == 1 ? 0.25 : 0.75);
    RF_vimpThreshold=q; uniform_tape[0]=u; uniform_n=1; uniform_at=0;
    Node *leaf=antiMembershipGeneric(1,&root,1,1,(double **)x);
    printf("anti,%g,%g,%u,%u\n",q,u,leaf->id,uniform_at);
  }
  /* Global factor mask test: unset level 2 must route right. */
  uint mask[2]={0, (1u<<0)|(1u<<2)|(1u<<3)};
  for(uint level=1; level<=4; ++level)
    printf("factor,%u,%d,0,0\n",level,(int)splitOnFactor(level,mask));
  return 0;
}
"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(prefix + functions + suffix)
    executable = OUT.with_suffix("")
    subprocess.run(
        ["cc", "-std=c99", "-O2", str(OUT), "-o", str(executable)], check=True, timeout=30
    )
    result = subprocess.run(
        [str(executable)], capture_output=True, text=True, check=True, timeout=10
    )
    FIXTURE.write_text(result.stdout)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    divisor = 1024**2 if sys.platform == "darwin" else 1024
    print(
        json.dumps(
            {
                "fixture": str(FIXTURE.relative_to(ROOT)),
                "elapsed_seconds": time.monotonic() - start,
                "parent_peak_mib": usage.ru_maxrss / divisor,
                "child_peak_mib": children.ru_maxrss / divisor,
                "swaps": usage.ru_nswap + children.ru_nswap,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
