"""Reproduce MULTINOMPOW native comparisons with declared source repairs.

Original sources stay under ignored research/raw. This tool never modifies them.
"""

import hashlib
import json
import subprocess
from pathlib import Path


DRIVER = """program reference
use mp_struct_mod
use mp_setup_mod
use power_mod
implicit none
integer i,j
read(*,*) n,p,n_pa,n_sig,n_ss
allocate(p0(p),pa(n_pa,p),sig(n_sig))
read(*,*) p0
 do i=1,n_pa
 read(*,*) pa(i,:)
 end do
read(*,*) sig
cs=.true.
lr=.true.
call setup_mp_power
! Correct empty rejection regions at the driver boundary, in both profiles.
where(cs_sig < 0)
 cs_sig=0
 cs_crit=huge(1d0)
end where
where(lr_sig < 0)
 lr_sig=0
 lr_crit=huge(1d0)
end where
cs_crit_min=minval(cs_crit)
lr_crit_min=minval(lr_crit)
call calculate_power
write(*,'(ES26.17E3)') sum(point_prob)
do i=1,n_sig
 write(*,'(4ES26.17E3)') cs_sig(i),cs_crit(i),lr_sig(i),lr_crit(i)
 do j=1,n_pa
 write(*,'(2ES26.17E3)') power_cs(j,i),power_lr(j,i)
 end do
end do
end program reference
"""


def main():
    from math import comb

    source = Path("research/raw/MULTINOMPOW/source/source/multinom_pow.1.0/source").resolve()
    archive = Path("research/raw/MULTINOMPOW/MULTINOMPOW _V1.tar.gz")
    work = Path("research/raw/reference/multinompow").resolve()
    names = [
        "gamma_mod",
        "log_factorial_mod",
        "summation_mod",
        "mp_struct_mod",
        "update_partition_mod",
        "perm_sort_array_mod",
        "mp_setup_mod",
        "power_mod",
    ]
    paths = [source / f"{name}.f90" for name in names]
    original = (source / "mp_setup_mod.f90").read_text()
    anchor = "        point(p) = n\n"
    if original.count(anchor) != 1:
        raise RuntimeError("unexpected source: first-point initialization anchor changed")
    repaired = original.replace(
        anchor, anchor + "        point_prob(1) = prob_point(point,log_p0)\n"
    )
    for name in ("chi_sq", "log_multinom_con"):
        if repaired.count(f"REAL :: {name}\n") != 1:
            raise RuntimeError(f"unexpected source declaration: {name}")
    provenance = {
        "archive_url": "https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/"
        "MULTINOMPOW/MULTINOMPOW%20_V1.tar.gz",
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "compiler": subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "driver": DRIVER,
        "profiles": {},
        "limitations": "Patched references, not unmodified native results. Alpha one is excluded; "
        "the native set_crit loop omits the final tie group. Driver runs one study per process, "
        "avoiding the native repeated-deallocation defect.",
    }
    cases = []
    for profile in ("initialized_single", "corrected_double"):
        directory = work / profile
        directory.mkdir(parents=True, exist_ok=True)
        adapted = directory / "mp_setup_mod.f90"
        text = repaired
        repairs = [
            "Initialize point_prob(1) after point(p)=n in traverse_points.",
            "Driver translates unset negative size/critical sentinels into size zero and "
            "huge positive critical values before calculate_power.",
        ]
        if profile == "corrected_double":
            for name in ("chi_sq", "log_multinom_con"):
                text = text.replace(f"REAL :: {name}\n", f"REAL (dpkind) :: {name}\n")
            repairs.append("Promote chi_sq and log_multinom_con return types to REAL(dpkind).")
        adapted.write_text(text)
        driver = directory / "driver.f90"
        driver.write_text(DRIVER)
        executable = directory / "reference"
        selected = [adapted if p.stem == "mp_setup_mod" else p for p in paths]
        command = [
            "gfortran",
            "-O0",
            "-ffp-contract=off",
            "-fcheck=all",
            *map(str, selected),
            str(driver),
            "-o",
            str(executable),
        ]
        subprocess.run(command, cwd=directory, check=True, capture_output=True, text=True)
        provenance["profiles"][profile] = {
            "repairs": repairs,
            "build_command": command,
            "adapted_setup_sha256": hashlib.sha256(adapted.read_bytes()).hexdigest(),
        }
        for n in (3, 7, 20, 60, 999, 1000, 1001):
            for null in (
                [0.5, 0.5],
                [0.2, 0.8],
                [1 / 3] * 3,
                [0.2, 0.3, 0.5],
                [0.1, 0.2, 0.3, 0.4],
            ):
                k = len(null)
                if n > 60 and k > 2:
                    continue
                alternatives = [null, list(reversed(null)), [1.0] + [0.0] * (k - 1)]
                # Original probability evaluation uses log(p), so use positive alternatives.
                alternatives[-1] = [0.9] + [0.1 / (k - 1)] * (k - 1)
                alpha = [0.000013, 0.011, 0.051, 0.203, 0.801]
                size = comb(n + k - 1, k - 1)
                lines = [
                    f"{n} {k} {len(alternatives)} {len(alpha)} {size}",
                    " ".join(map(str, null)),
                    *(" ".join(map(str, row)) for row in alternatives),
                    " ".join(map(str, alpha)),
                ]
                output = subprocess.check_output(
                    [str(executable)],
                    input="\n".join(lines) + "\n",
                    text=True,
                    cwd=directory,
                    timeout=60,
                )
                numbers = [float(x) for x in output.split()]
                if len(numbers) != 1 + len(alpha) * (4 + 2 * len(alternatives)):
                    raise RuntimeError(f"unexpected native output: {output}")
                rows, offset = [], 1
                for _ in alpha:
                    values = numbers[offset : offset + 4]
                    offset += 4
                    power = []
                    for _ in alternatives:
                        power.append(numbers[offset : offset + 2])
                        offset += 2
                    rows.append(
                        {
                            "size": [values[0], values[2]],
                            "critical": [None if x > 1e300 else x for x in (values[1], values[3])],
                            "power": power,
                        }
                    )
                cases.append(
                    {
                        "profile": profile,
                        "n": n,
                        "null": null,
                        "alternatives": alternatives,
                        "alpha": alpha,
                        "null_mass": numbers[0],
                        "rows": rows,
                    }
                )
    destination = Path("tests/fixtures/multinompow_native.json")
    destination.write_text(json.dumps({"provenance": provenance, "cases": cases}, indent=2) + "\n")
    print(f"Wrote {len(cases)} native studies to {destination}")


if __name__ == "__main__":
    main()
