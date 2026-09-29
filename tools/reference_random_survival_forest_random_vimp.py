#!/usr/bin/env python3
"""Extract and call pinned randomForestSRC's unchanged generic route kernel."""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import re
import subprocess
import sys
import tempfile
from pathlib import Path

PIN = "b4d099e262423362a8872c13c468e6dbe2f9e9da"
HASH = "e9c6e896c4f93c6eb1b85bdf37992964d74376ea"
FUNCTION = "randomMembershipGeneric"


def git_blob_hash(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def extract_function(source: str, name: str) -> str:
    match = re.search(rf"^Node \*{name}\s*\(", source, re.M)
    if match is None:
        raise ValueError(f"source function not found: {name}")
    brace = source.index("{", match.start())
    depth = 0
    for end in range(brace, len(source)):
        if source[end] == "{":
            depth += 1
        elif source[end] == "}":
            depth -= 1
            if depth == 0:
                return source[match.start() : end + 1] + "\n"
    raise ValueError(f"unbalanced source function: {name}")


def build_lib(root: Path) -> ctypes.CDLL:
    srcpath = root / "research/raw/randomForestSRC/src/randomForestSRC.c"
    data = srcpath.read_bytes()
    if git_blob_hash(data) != HASH:
        raise ValueError(f"pinned source hash mismatch: {srcpath}")
    source = data.decode()
    extracted = extract_function(source, FUNCTION)
    header = r"""
#include <stdint.h>
#include <stddef.h>
typedef unsigned int uint;
#define LEFT 1
#define RIGHT 2
#define TRUE 1
#define FALSE 0
typedef struct SplitInfo { uint randomVar[2]; char ordinary; } SplitInfo;
typedef struct Node Node;
struct Node { Node *left, *right; SplitInfo *splitInfo; uint repMembrSize, nodeID; };
static double RF_vimpThreshold;
static char RF_importanceFlag[8];
static double fixed_alpha;
static unsigned int draw_count;
static double ran1D(uint treeID) { (void)treeID; draw_count++; return fixed_alpha; }
static char getDaughterPolarity(int ignored, SplitInfo *info, uint individual, double **xArray) {
  (void)ignored; (void)individual; (void)xArray; return info->ordinary;
}
"""
    wrapper = r"""
unsigned int route_one(unsigned int n, unsigned int left_n,
                       unsigned int split_feature, unsigned int target_feature,
                       unsigned int ordinary_branch, unsigned int group_flag,
                       double threshold, double alpha, int terminal_root) {
  Node root={0}, left={0}, right={0}; SplitInfo split={0};
  draw_count=0; fixed_alpha=alpha; RF_vimpThreshold=threshold;
  for (unsigned int i=0;i<8;i++) RF_importanceFlag[i]=0;
  RF_importanceFlag[split_feature]=(char)group_flag;
  root.repMembrSize=n; root.nodeID=terminal_root ? 103u : 0u; split.randomVar[1]=split_feature;
  split.ordinary=(char)ordinary_branch;
  if (!terminal_root) {
    left.repMembrSize=left_n; left.nodeID=101;
    right.repMembrSize=n-left_n; right.nodeID=102;
    root.left=&left; root.right=&right; root.splitInfo=&split;
  }
  Node *result=randomMembershipGeneric(1,&root,1,target_feature,NULL);
  return result->nodeID*10000u+draw_count;
}
"""
    with tempfile.TemporaryDirectory(prefix="rsf-random-vimp-") as temp_dir:
        temp = Path(temp_dir)
        cfile = temp / "native_kernel.c"
        cfile.write_text(header + extracted + wrapper)
        libpath = temp / ("native_kernel.dylib" if sys.platform == "darwin" else "native_kernel.so")
        kind = "-dynamiclib" if sys.platform == "darwin" else "-shared"
        subprocess.run(
            ["cc", "-O0", "-fPIC", kind, str(cfile), "-o", str(libpath)],
            check=True,
            timeout=45,
        )
        lib = ctypes.CDLL(str(libpath))
        fn = lib.route_one
        fn.argtypes = [ctypes.c_uint] * 6 + [ctypes.c_double] * 2 + [ctypes.c_int]
        fn.restype = ctypes.c_uint
        return lib


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    default_root = Path(__file__).resolve().parents[1]
    ap.add_argument("--root", type=Path, default=default_root)
    ap.add_argument(
        "--cases",
        type=Path,
        default=default_root / "tests/fixtures/random-survival-forest-random-routing.csv",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=default_root / "tests/fixtures/random-survival-forest-random-routing-native.csv",
    )
    args = ap.parse_args()
    lib = build_lib(args.root)
    rows = []
    with args.cases.open(newline="") as f:
        for row in csv.DictReader(f):
            packed = lib.route_one(
                int(row["node_count"]),
                int(row["left_count"]),
                int(row["split_feature"]),
                int(row["target_feature"]),
                int(row["ordinary_branch"]),
                int(row["group_flag"]),
                float(row["threshold"]),
                float(row["alpha"]),
                int(row["scope"] == "terminal_node"),
            )
            actual_node, draws = divmod(int(packed), 10000)
            row["actual_terminal"] = str(actual_node)
            row["actual_draws"] = str(draws)
            row["match"] = (
                "true"
                if (
                    actual_node == int(row["expected_terminal"])
                    and draws == int(row["expected_draws"])
                )
                else "false"
            )
            rows.append(row)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    mismatches = [r["case_id"] for r in rows if r["match"] != "true"]
    print(f"source={PIN} function={FUNCTION} cases={len(rows)} mismatches={mismatches}")


if __name__ == "__main__":
    main()
