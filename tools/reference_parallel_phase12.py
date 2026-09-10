"""Compare Python replay with original C control flow; numerical/RNG adapters are explicit."""

import ctypes as ct
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.special import betainc

from mdanderson_stats.beta_binomial import BetaBinomialPosterior
from mdanderson_stats.beta_comparison import compare_beta_binomial
from mdanderson_stats.parallel_phase12 import parallel_phase12_replay

root = Path.cwd()
source = root / "research/raw/P12Xuelin/extracted/SwatiBiswasCode"
with tempfile.TemporaryDirectory(prefix="p12-reference-") as temp:
    work = Path(temp)
    header = (
        (source / "def.h")
        .read_text()
        .replace('"/home/biswas/ranlib.c/src/ranlib.h"', '"ranlib.h"')
        .replace('"/home/biswas/library/lib.h"', '"lib.h"')
    )
    (work / "def.h").write_text(header)
    for name in ["main_4arms_nobugs.c", "functions_4arms_nobugs.c", "ranlib.h", "lib.h"]:
        (work / name).write_bytes((source / name).read_bytes())
    (work / "adapter.c").write_text("""
#include <stdlib.h>
#include <stdio.h>
static double (*cdf_callback)(double,double,double);
static double (*order_callback)(double,double,double,double);
static long (*binomial_callback)(long,float);
static double (*uniform_callback)(void);
void initialize(void *a, void *b, void *c, void *d) {
 cdf_callback=a;order_callback=b;binomial_callback=c;uniform_callback=d;
}
double betai(double a,double b,double x){ return cdf_callback(a,b,x); }
double sum(double *x,int n){ double s=0;for(int i=0;i<n;i++)s+=x[i];return s; }
long ignbin(long n,float p){return binomial_callback(n,p);}
float genunf(float a,float b){return a+(b-a)*uniform_callback();}
void setall(long a,long b){}
int p12system(const char *unused){
 double a,b,c,d;FILE *f=fopen("inputR.txt","r");
 if(!f || fscanf(f,"%lf %lf %lf %lf",&a,&b,&c,&d)!=4)abort();
 fclose(f);double p=order_callback(a,b,c,d);
 f=fopen("outputR","w");fprintf(f,"p %.17g\\n",p);fclose(f);return 0;
}
""")
    subprocess.run(
        [
            "cc",
            "-shared",
            "-fPIC",
            "-std=c99",
            "-Wno-format",
            "-Dmain=source_main",
            "-Dsystem=p12system",
            "main_4arms_nobugs.c",
            "functions_4arms_nobugs.c",
            "adapter.c",
            "-o",
            "source.dylib",
        ],
        cwd=work,
        check=True,
    )
    lib = ct.CDLL(str(work / "source.dylib"))
    cdf_type = ct.CFUNCTYPE(ct.c_double, ct.c_double, ct.c_double, ct.c_double)
    order_type = ct.CFUNCTYPE(ct.c_double, ct.c_double, ct.c_double, ct.c_double, ct.c_double)
    binom_type = ct.CFUNCTYPE(ct.c_long, ct.c_long, ct.c_float)
    uniform_type = ct.CFUNCTYPE(ct.c_double)
    comparisons = {}

    def order(a, b, c, d):
        key = (a, b, c, d)
        if key not in comparisons:
            comparisons[key] = float(
                compare_beta_binomial(
                    BetaBinomialPosterior(c, d), BetaBinomialPosterior(a, b)
                ).treatment_greater
            )
        return comparisons[key]

    q = np.array([0.04, 0.09, 0.16, 0.25])
    records = []
    pending = []
    rng = np.random.default_rng(8501)

    def binomial(n, p):
        value = int(rng.binomial(n, p))
        if not pending:
            arm = int(np.argmin(abs(q - p)))
            assert abs(q[arm] - p) < 1e-7
            pending.extend([arm, n, value])
        else:
            arm, size, tox = pending
            assert size == n
            records.extend([(arm, int(i < tox), int(i < value)) for i in range(n)])
            pending.clear()
        return value

    callbacks = (
        cdf_type(betainc),
        order_type(order),
        binom_type(binomial),
        uniform_type(lambda: float(rng.random())),
    )
    lib.initialize.argtypes = [ct.c_void_p] * 4
    lib.initialize(*callbacks)
    lib.source_main.argtypes = [ct.c_int, ct.POINTER(ct.c_char_p)]
    (work / "in_seed").write_text("8501\n")
    results = []
    os.chdir(work)
    try:
        for i in range(24):
            q = np.array([0.04, 0.09, 0.16, 0.25]) if i < 12 else np.array([0.18, 0.29, 0.42, 0.55])
            p = np.array([0.1, 0.2, 0.35, 0.5]) if i % 3 else np.array([0.01, 0.02, 0.03, 0.04])
            records.clear()
            pending.clear()
            ct.c_int.in_dll(lib, "tot_num_pat").value = 0
            ct.c_double.in_dll(lib, "prob_gt02_best").value = 0
            ct.c_int.in_dll(lib, "best_treat").value = 0
            adm = (ct.c_int * 4).in_dll(lib, "adm")
            for j in range(4):
                adm[j] = 0
            (work / "scenario").write_text(" ".join(map(str, np.r_[p, q])))
            output = work / f"output-{i}"
            args = [b"source", b"scenario", str(output).encode(), b"seeds"]
            argv = (ct.c_char_p * 4)(*args)
            assert lib.source_main(4, argv) == 0
            values = np.loadtxt(output)
            result = parallel_phase12_replay(records)
            expected = int(values[0]) - 1
            assert result.selected == (None if expected < 0 else expected), (i, result, values)
            native = values[3:].reshape(4, 4)
            np.testing.assert_array_equal(result.treated, native[:, 0])
            np.testing.assert_array_equal(result.toxicities, native[:, 1])
            np.testing.assert_array_equal(result.responses, native[:, 2])
            assert result.phase == "complete"
            results.append(
                {
                    "trial": i,
                    "enrollment": len(records),
                    "selected": result.selected,
                    "reason": result.reason,
                }
            )
    finally:
        os.chdir(root)
    out = {
        "archive_url": "https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/P12Xuelin/P12Xuelin_V1.0_.zip",
        "archive_sha256": hashlib.sha256(
            (root / "research/raw/P12Xuelin/archive.zip").read_bytes()
        ).hexdigest(),
        "native_control_flow": (
            "Original C main and functions; absolute include paths rewritten. "
            "CDF, external R comparison and RNG replaced by explicit callbacks; "
            "not an independent numerical comparison."
        ),
        "seed": 8501,
        "trials": results,
        "unique_beta_comparisons": len(comparisons),
    }
    (root / "docs/parallel-phase12-reference.json").write_text(json.dumps(out, indent=2) + "\n")
    np.savetxt(
        root / "research/raw/P12Xuelin/beta-comparisons.csv",
        np.array([(*key, value) for key, value in comparisons.items()]),
        delimiter=",",
    )
    print(json.dumps(out, indent=2))
