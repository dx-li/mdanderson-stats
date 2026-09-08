"""Compile MULTI's archived numerical routines and record reference results.

The desktop main program is removed at a program-unit boundary. The S library's
SCHWED routine is copied into the local build with only its name changed, to
distinguish its extra variance output. OSFIT and CDFBET are extracted unchanged.
Original numerical code is not bundled.
"""

import hashlib
import json
import re
import subprocess
from pathlib import Path

import numpy as np
from reference_numerics import BUILD, RAW, run

SOURCE = RAW / "MULTI/source"


def build() -> Path:
    directory = BUILD / "multi"
    directory.mkdir(parents=True, exist_ok=True)
    original = (SOURCE / "multi3.f").read_text()
    start = original.index("      PROGRAM MAIN")
    end = original.index("      DOUBLE PRECISION FUNCTION MIXBET", start)
    library3 = directory / "multi3-library.f"
    library3.write_text(original[:start] + original[end:])
    s_source = (RAW / "MULTI/S/multi-1.0/multi.f").read_text()
    start = s_source.index("      SUBROUTINE SCHWED(")
    end_match = re.search(r"^      END\s*$", s_source[start:], flags=re.MULTILINE)
    if end_match is None:
        raise ValueError("SCHWED program unit not found")
    s_schweder = directory / "s-schweder.f"
    s_schweder.write_text(
        s_source[start : start + end_match.end()].replace(
            "SUBROUTINE SCHWED(", "SUBROUTINE S_SCHWED(", 1
        )
        + "\n"
    )
    start = s_source.index("      SUBROUTINE OSFIT(")
    end_match = re.search(r"^      END\s*$", s_source[start:], flags=re.MULTILINE)
    if end_match is None:
        raise ValueError("OSFIT program unit not found")
    s_osfit = directory / "s-osfit.f"
    s_osfit.write_text(s_source[start : start + end_match.end()] + "\n")
    start = s_source.index("      SUBROUTINE cdfbet(")
    end_match = re.search(r"^      END\s*$", s_source[start:], flags=re.MULTILINE)
    if end_match is None:
        raise ValueError("CDFBET program unit not found")
    s_cdfbet = directory / "s-cdfbet.f"
    s_cdfbet.write_text(s_source[start : start + end_match.end()] + "\n")
    driver = directory / "driver.f90"
    driver.write_text("""program multi_reference
  implicit none
  integer :: mode,n,i,number,status,length
  integer, external :: shl,shc
  double precision :: alpha,nulls,beta,variance
  double precision, allocatable :: x(:),p(:),q(:)
  integer, allocatable :: counts(:)
  read(*,*) mode,n,alpha,nulls
  allocate(x(n),p(n),q(n),counts(n))
  read(*,*) x
  select case(mode)
  case(1)
    call osbf(x,p,n)
  case(2)
    call ossd(x,p,n)
  case(3)
    call sdbf(x,p,n)
  case(4)
    call sdsd(x,p,n)
  case(5)
    call sdfn(x,p,n)
  case(6)
    call suhc(x,p,n)
  case(7)
    call suhm(x,p,n)
  case(8)
    call susm(x,p,n)
  case(9)
    call surm(p,n,alpha)
  case(10,11)
    if (mode == 10) then
      number=shl(x,n,alpha,nulls)
    else
      number=shc(x,n,alpha,nulls)
    end if
    write(*,*) number
    stop
  case(12,13)
    variance=0d0
    if (mode == 12) then
      call schwed(x,n,alpha,p,counts,length,beta,status)
    else
      call s_schwed(x,n,alpha,p,counts,length,beta,variance,status)
    end if
    write(*,'(2i8,2es26.17)') status,length,beta,variance
    do i=1,length
      write(*,'(es26.17,i8)') p(i),counts(i)
    end do
    stop
  case(14)
    call osfit(x,p,q,beta,n)
    write(*,'(es26.17)') beta
    do i=1,n
      write(*,'(2es26.17)') p(i),q(i)
    end do
    stop
  end select
  do i=1,n
    write(*,'(es26.17)') p(i)
  end do
end program
""")
    executable = (directory / "multi").resolve()
    sources = [
        SOURCE / "multi1.f",
        SOURCE / "multi2.f",
        library3,
        SOURCE / "multi4.f",
        SOURCE / "maccon.f",
        SOURCE / "lterm.f",
        s_schweder,
        s_osfit,
        s_cdfbet,
        driver,
    ]
    result = subprocess.run(
        [
            "gfortran",
            "-O2",
            "-std=legacy",
            "-fallow-argument-mismatch",
            *map(str, sources),
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
    )
    (directory / "build.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f"Reference build failed; see {directory / 'build.log'}")
    return executable


def evaluate(executable: Path, mode: int, values: list[float], alpha=0.05, nulls=0) -> str:
    return run(
        executable,
        f"{mode} {len(values)} {alpha} {nulls}\n" + " ".join(map(str, sorted(values))) + "\n",
    )


def main() -> None:
    executable = build()
    families = [
        [0.05],
        [0.001, 0.04, 0.02, 0.8, 0.15],
        [0, 1, 0.05, 0.05, 0.0000001, 0.2, 0.8, 0.01],
        [float(v) for v in (SOURCE / "pvals.txt").read_text().split()],
    ]
    names = [
        "bonferroni",
        "sidak",
        "holm",
        "holm-sidak",
        "finner",
        "hochberg",
        "multi-hommel",
        "simes",
    ]
    adjustments = []
    for family in families:
        outputs = {}
        for mode, name in enumerate(names, start=1):
            outputs[name] = list(map(float, evaluate(executable, mode, family).split()))
        adjustments.append({"pvalues": family, "sorted_adjusted": outputs})
    rom = []
    for n in (1, 2, 3, 5, 20, 50, 150):
        for alpha in (0.01, 0.05, 0.1, 0.5, 0.9):
            thresholds = list(map(float, evaluate(executable, 9, [0.5] * n, alpha).split()))
            rom.append({"n": n, "alpha": alpha, "thresholds": thresholds})
    sharpened = []
    for family in families:
        for nulls in (0, 0.9, 1, 3.8, len(family), len(family) + 5):
            for alpha in (0.05, 0.2):
                sharpened.append(
                    {
                        "pvalues": family,
                        "null_estimate": nulls,
                        "alpha": alpha,
                        "holm": int(evaluate(executable, 10, family, alpha, nulls)),
                        "hochberg": int(evaluate(executable, 11, family, alpha, nulls)),
                    }
                )
    fits = []
    fit_families = [
        families[-1],
        [i / 21 for i in range(1, 21)],
        [0.01] * 10 + [0.1, 0.2, 0.4, 0.6, 0.8, 0.9],
        [1e-5 * i for i in range(1, 16)] + [0.4, 0.6, 0.8, 0.9],
        [0.1, 0.2, 0.3, 0.99],
    ]
    for family in fit_families:
        for alpha in (0.05, 0.2, 0.5):
            records = {}
            for mode, name in [(12, "desktop"), (13, "splus")]:
                rows = evaluate(executable, mode, family, alpha).splitlines()
                status, length, beta, variance = rows[0].split()
                records[name] = {
                    "status": int(status),
                    "null_estimate": float(beta),
                    "prediction_variance": float(variance),
                    "one_minus_p": [float(row.split()[0]) for row in rows[1:]],
                    "upper_counts": [int(row.split()[1]) for row in rows[1:]],
                }
                if len(rows) != int(length) + 1:
                    raise RuntimeError("Unexpected Schweder output length")
            fits.append({"pvalues": family, "alpha": alpha, "reference": records})
    bootstrap_values = np.linspace(0.02, 0.98, 40)
    generator = np.random.default_rng(126)
    bootstrap_estimates = []
    for attempt in range(1, 101):
        resampled = generator.choice(bootstrap_values, size=40, replace=True).tolist()
        status, _, estimate, _ = evaluate(executable, 13, resampled).splitlines()[0].split()
        if int(status) == 0:
            bootstrap_estimates.append(float(estimate))
        if len(bootstrap_estimates) == 50:
            break
    else:
        raise RuntimeError("Reference bootstrap did not produce enough fits")
    result = {
        "source_url": "https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/MULTI/MULTI_V1.tar.gz",
        "sha256": hashlib.sha256((RAW / "MULTI/MULTI_V1.tar.gz").read_bytes()).hexdigest(),
        "compiler": subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
        "flags": ["-O2", "-std=legacy", "-fallow-argument-mismatch"],
        "adjustments": adjustments,
        "rom": rom,
        "sharpened": sharpened,
        "schweder": fits,
        "bootstrap": {
            "pvalues": bootstrap_values.tolist(),
            "samples": 50,
            "seed": 126,
            "attempts": attempt,
            "estimates": bootstrap_estimates,
            "sampling": "NumPy Generator; each resample fitted by archived S SCHWED",
        },
    }
    diagnostics = []
    for family in families + [
        [0.8, 0.9, 0.95],
        [1, 1, 1],
        [0, 0, 0],
        [1e-12, 1e-9, 0.3, 0.5, 1],
        [0.2, 0.3, 0.4],
        [i / 41 for i in range(1, 41)],
    ]:
        rows = evaluate(executable, 14, family).splitlines()
        diagnostics.append(
            {
                "pvalues": family,
                "combined_score": float(rows[0]),
                "sorted_cumulative": [float(row.split()[0]) for row in rows[1:]],
                "sorted_legacy_transformed_cdf": [float(row.split()[1]) for row in rows[1:]],
            }
        )
    result["order_statistics"] = diagnostics
    Path("tests/fixtures/multi.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(
        f"Recorded {len(adjustments) * 8} adjustments, {len(rom)} Rom thresholds, "
        f"{len(sharpened) * 2} sharpened tests, {len(fits) * 2} Schweder fits "
        f"and {len(diagnostics)} order-statistic diagnostics"
    )


if __name__ == "__main__":
    main()
