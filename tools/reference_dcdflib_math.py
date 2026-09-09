"""Direct C/F77 evidence for the 31 mathematical support contracts."""

import hashlib
import itertools
import json
import math
import subprocess
import tarfile
import tempfile
from pathlib import Path

from audit_cdflib90 import ARCHIVE_SHA256

# Native argument names refer to the uniform input record below.
FUNCTIONS = {
    "alnrel": "x",
    "rexp": "x",
    "rlog": "x",
    "rlog1": "x",
    "alngam": "x",
    "gamln": "x",
    "gamln1": "x",
    "gam1": "x",
    "Xgamm": "x",
    "psi": "x",
    "erf1": "x",
    "algdiv": "a,b",
    "bcorr": "a,b",
    "betaln": "a,b",
    "gsumln": "a,b",
    "rcomp": "a,x",
    "erfc1": "k,x",
    "esum": "k,x",
    "exparg": "k",
    "apser": "a,b,x,eps",
    "fpser": "a,b,x,eps",
    "bpser": "a,b,x,eps",
    "basym": "a,b,lam,eps",
    "bfrac": "a,b,x,y,lam,eps",
    "brcomp": "a,b,x,y",
    "brcmp1": "k,a,b,x,y",
    "bup": "a,b,x,y,k,eps",
}
SUBROUTINES = {
    "bgrat": "a,b,x,y,v,eps,status",
    "bratio": "a,b,x,y,v,q,status",
    "grat1": "a,x,r,v,q,eps",
    "gratio": "a,x,v,q,k",
}
OPS = list(FUNCTIONS) + list(SUBROUTINES)
ALIASES = {"Xgamm": "gamma", "erf1": "erf"}
FIELDS = ["a", "b", "x", "y", "lam", "eps", "w", "r"]


def drivers():
    c, f = [], []
    for i, op in enumerate(OPS, 1):
        args = (FUNCTIONS | SUBROUTINES)[op].split(",")
        ccall = op + "(" + ",".join("&" + a for a in args) + ")"
        fcall = ALIASES.get(op, op) + "(" + ",".join(args) + ")"
        c.append(f"case {i}: " + ("v=" if op in FUNCTIONS else "") + ccall + "; break;")
        f.append(f"case({i})\n" + ("v=" if op in FUNCTIONS else "call ") + fcall)
    csource = (
        """#include <stdio.h>
#include "cdflib.h"
int main(void) {
 int op,k,status=0; double a,b,x,y,lam,eps,w,r,v,q=0;
 if(scanf("%d %d %lf %lf %lf %lf %lf %lf %lf %lf",&op,&k,&a,&b,&x,&y,&lam,&eps,&w,&r)!=10)return 2;
 v=w;
 switch(op){"""
        + "".join(c)
        + """default:return 2;}
 printf("%d %.17g %.17g\\n",status,v,q);
 return 0;
}"""
    )
    fsource = (
        """program probe
implicit none
integer op,k,status
real(8) a,b,x,y,lam,eps,w,r,v,q
"""
        + "".join("real(8),external :: " + ALIASES.get(op, op) + "\n" for op in FUNCTIONS)
        + """read(*,*) op,k,a,b,x,y,lam,eps,w,r
v=w
q=0
status=0
select case(op)
"""
        + "\n".join(f)
        + """
end select
write(*,'(I5,2ES26.17E3)') status,v,q
end program probe
"""
    )
    return {"c": csource, "fortran": fsource}


def cases():
    rows = []

    def add(op, tag="ordinary", **kw):
        row = dict(
            op=op, tag=tag, k=0, a=1.0, b=1.0, x=0.5, y=0.5, lam=0.0, eps=5e-15, w=0.0, r=0.0
        )
        row.update(kw)
        if "x" in kw and "y" not in kw:
            row["y"] = 1 - row["x"]
        if op == "bfrac" and "lam" not in kw:
            row["lam"] = (row["a"] + row["b"]) * row["y"] - row["b"]
        rows.append(row)

    grids = {
        "alnrel": [-0.999, -0.5, -0.375, -0.1, -1e-10, 0, 1e-10, 0.1, 0.375, 1, 10],
        "rexp": [-700, -0.15, -1e-10, 0, 1e-10, 0.15, 1, 700],
        "rlog": [0.1, 0.61, 0.82, 1, 1.18, 1.57, 2, 10],
        "rlog1": [-0.99, -0.39, -0.18, -1e-10, 0, 1e-10, 0.18, 0.57, 1, 10],
        "alngam": [0.001, 0.1, 0.8, 1, 1.25, 2, 2.25, 3, 6, 8, 12, 100],
        "gamln": [0.001, 0.1, 0.8, 1, 1.25, 2, 2.25, 3, 6, 8, 12, 100],
        "gamln1": [-0.2, -0.1, -1e-10, 0, 1e-10, 0.6, 1, 1.25],
        "gam1": [-0.5, -0.1, -1e-10, 0, 1e-10, 0.5, 1, 1.5],
        "Xgamm": [-10.5, -3.5, -0.5, 0.1, 0.5, 1, 2, 3.5, 15, 100, 170],
        "psi": [-10.5, -3.5, -0.5, 0.1, 0.5, 1, 2, 3.5, 15, 100, 170],
        "erf1": [-10, -5.8, -4, -0.5, -1e-10, 0, 1e-10, 0.5, 4, 5.8, 10],
    }
    for op, xs in grids.items():
        for x in xs:
            add(op, x=x)
    for k, x in itertools.product([0, 1, -2], [-5.8, -4, -0.5, 0, 0.5, 4, 5.8, 10, 26]):
        add("erfc1", k=k, x=x)
    for k in [-700, -1, 0, 1, 700]:
        add("esum", k=k, x=-k + 0.125)
    for k in [-2, 0, 1, 2]:
        add("exparg", k=k)
    for a, b in itertools.product([0, 0.1, 1, 10, 100], [8, 10, 100]):
        add("algdiv", a=a, b=b)
    for a, b in itertools.product([8, 10, 100], repeat=2):
        add("bcorr", a=a, b=b)
    for a, b in itertools.product([0.1, 0.5, 1, 2, 8, 100], repeat=2):
        add("betaln", a=a, b=b)
    for a, b in itertools.product([1, 1.2, 2], [1, 1.3, 2]):
        add("gsumln", a=a, b=b)
    for a in [0.1, 0.5, 1, 2, 20, 100]:
        for x in [0.1, 1, a, 2 * a]:
            add("rcomp", a=a, x=x)
    for a, x in itertools.product([0.1, 1, 20], [1e-6, 0.1, 0.5]):
        add("fpser", a=a, b=1e-20, x=x)
    for b, x in itertools.product([0.1, 1, 10], [1e-6, 0.01, 0.1]):
        add(
            "apser",
            tag="rounded_product" if b == 10 and x == 0.1 else "ordinary",
            a=1e-20,
            b=b,
            x=x,
        )
    for a, b, x in itertools.product([0.1, 1, 10], [0.1, 0.5, 1, 2], [0.01, 0.1]):
        add("bpser", a=a, b=b, x=x)
    for a, b, factor in itertools.product([2, 10, 100], [2, 10, 100], [0.5, 1]):
        add("bfrac", a=a, b=b, x=(a / (a + b)) * factor)
    for a, b, factor in itertools.product([15, 20, 1000], [15, 20, 1000], [0, 0.01, 0.03]):
        add("basym", a=a, b=b, lam=min(a, b) * factor)
    for a, b, x in itertools.product([0.1, 1, 2, 10], [0.1, 1, 2, 10], [0.1, 0.5, 0.9]):
        add("brcomp", a=a, b=b, x=x)
    for k, x in itertools.product([-700, -1, 0, 1, 700], [0.1, 0.5, 0.9]):
        add("brcmp1", k=k, a=2, b=3, x=x)
    for a, b, x, k in itertools.product([0.1, 1, 10], [0.1, 1, 10], [0.1, 0.5], [1, 2, 10]):
        add("bup", a=a, b=b, x=x, k=k)
    for a, b, x, w in itertools.product([15, 20, 100], [0.1, 0.5, 1], [0.8, 0.95], [0, 0.25]):
        add("bgrat", a=a, b=b, x=x, w=w)
    for a, b, x in itertools.product([0.1, 1, 2, 20], [0.1, 1, 2, 20], [0, 0.1, 0.5, 0.9, 1]):
        add("bratio", a=a, b=b, x=x)
    for a, b, x in [(0, 1, 0.5), (1, 0, 0.5), (0, 1, 1), (1, 0, 0)]:
        add("bratio", a=a, b=b, x=x)
    for a, x in itertools.product([0, 0.1, 0.5, 1], [0, 0.1, 1, 1.1, 10]):
        r = math.exp(a * math.log(x) - x - math.lgamma(a)) if a > 0 and x > 0 else 0.0
        add("grat1", a=a, x=x, r=r)
    for factor in [0, 0.5, 1]:
        add("grat1", a=0.1, x=2, r=math.exp(0.1 * math.log(2) - 2 - math.lgamma(0.1)) * factor)
    for a, x, k in itertools.product([0, 0.1, 0.5, 1, 2, 20, 100], [0, 0.1, 1, 5, 100], [0, 1, 2]):
        if a or x:
            add("gratio", a=a, x=x, k=k)
    # Explicit probes for source sentinels, numerical losses and invalid domains.
    for op, kw in [
        ("rlog1", dict(x=3e-162)),
        ("rlog1", dict(x=-3e-162)),
        ("erfc1", dict(k=0, x=27.2)),
        ("erfc1", dict(k=1, x=1e308)),
        ("gamln", dict(x=1e-309)),
        ("alngam", dict(x=1e-309)),
        ("Xgamm", dict(x=-172.5)),
        ("psi", dict(x=-2147483647.5)),
        ("algdiv", dict(a=1e-100, b=1e308)),
        ("bcorr", dict(a=1e308, b=1e308)),
        ("betaln", dict(a=1e308, b=1e308)),
        ("rcomp", dict(a=1e300, x=1e300)),
        ("bratio", dict(a=100, b=1e-20, x=0.001)),
        ("bgrat", dict(a=20, b=0.1, x=0.9, w=-0.25)),
        ("brcomp", dict(a=1e300, b=1e300, x=0.5)),
        ("basym", dict(a=1e300, b=1e300, lam=0)),
    ]:
        add(op, tag="repair", **kw)
    for op, kw in [
        ("alnrel", dict(x=-1)),
        ("rlog", dict(x=0)),
        ("rlog1", dict(x=-1)),
        ("gamln", dict(x=0)),
        ("Xgamm", dict(x=0)),
        ("psi", dict(x=0)),
        ("gratio", dict(a=0, x=0)),
    ]:
        add(op, tag="invalid", **kw)
    for kw in [
        dict(a=-1),
        dict(a=0, b=0),
        dict(x=-0.1, y=1),
        dict(x=0.5, y=-0.1),
        dict(x=0.5, y=0.6),
        dict(a=0, x=0),
        dict(b=0, x=1),
    ]:
        add("bratio", tag="invalid", **kw)
    return rows


def main():
    archive = Path("research/raw/CDFLIB90/CDFLIB90  _V90.tar.gz")
    if hashlib.sha256(archive.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        raise RuntimeError("Archive hash mismatch")
    profiles = {}
    sources = {}
    with tarfile.open(archive) as tar:
        for member in tar:
            if member.isfile() and (
                member.name.startswith("source/dcdflib.c/src/")
                or member.name.startswith("source/dcdflib.f/src/")
            ):
                stream = tar.extractfile(member)
                if stream is None:
                    raise RuntimeError("Missing native source")
                sources[member.name] = stream.read()
    ds = drivers()
    with tempfile.TemporaryDirectory(prefix="dcdflib-math-") as tmp:
        work = Path(tmp)
        for language in ["c", "fortran"]:
            prefix = "source/dcdflib.c/src/" if language == "c" else "source/dcdflib.f/src/"
            selected = {n: r for n, r in sources.items() if n.startswith(prefix)}
            for n, raw in selected.items():
                (work / Path(n).name).write_bytes(raw)
            driver_name = "driver.c" if language == "c" else "driver.f90"
            (work / driver_name).write_text(ds[language])
            flags = (
                ["cc", "-std=gnu89", "-O0", "-ffp-contract=off"]
                if language == "c"
                else ["gfortran", "-std=legacy", "-O0", "-ffp-contract=off", "-fcheck=all"]
            )
            link = ["-lm"] if language == "c" else []
            inputs = sorted(Path(n).name for n in selected if not n.endswith(".h"))
            subprocess.run(
                [*flags, *inputs, driver_name, *link, "-o", "reference"],
                cwd=work,
                check=True,
                capture_output=True,
                timeout=60,
            )
            records = []
            for row in cases():
                payload = (
                    " ".join(
                        map(str, [OPS.index(row["op"]) + 1, row["k"], *[row[f] for f in FIELDS]])
                    )
                    + "\n"
                )
                try:
                    run = subprocess.run(
                        [str(work / "reference")],
                        input=payload,
                        text=True,
                        capture_output=True,
                        timeout=3,
                    )
                except subprocess.TimeoutExpired:
                    records.append(dict(**row, outcome="timeout", timeout_seconds=3))
                    continue
                if run.returncode:
                    records.append(
                        dict(
                            **row,
                            outcome="process_error",
                            returncode=run.returncode,
                            stderr=run.stderr,
                        )
                    )
                    continue
                fields = run.stdout.split()
                if len(fields) != 3:
                    raise RuntimeError(f"Unexpected output {run.stdout!r}")
                vals = [float(v) for v in fields[1:]]
                records.append(
                    dict(
                        **row,
                        outcome="completed",
                        status=int(fields[0]),
                        values=[v if math.isfinite(v) else str(v) for v in vals],
                    )
                )
            profiles[language] = dict(
                source_hashes={n: hashlib.sha256(r).hexdigest() for n, r in selected.items()},
                driver_sha256=hashlib.sha256(ds[language].encode()).hexdigest(),
                flags=flags,
                link_flags=link,
                compiler=subprocess.check_output([flags[0], "--version"], text=True).splitlines()[
                    0
                ],
                cases=records,
            )
            print(language, len(records), "cases", flush=True)
    Path("tests/fixtures/dcdflib_math.json").write_text(
        json.dumps(dict(archive_sha256=ARCHIVE_SHA256, profiles=profiles), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
