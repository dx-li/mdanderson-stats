"""Native SEQBIN FIND_STOP calibration using unchanged source routines."""

import hashlib
import json
import re
import subprocess
from pathlib import Path


def main():
    source = Path("research/raw/SEQBIN/source/seqbin/SOURCE").resolve()
    original = (source / "design_properties_module.f90").read_text()
    directory = Path("research/raw/reference/seqbin_calibration").resolve()
    directory.mkdir(parents=True, exist_ok=True)
    blocks = []
    for name in [
        "bayesian_design",
        "calculate_design",
        "calculate_properties",
        "calculate_design_properties",
        "convolve_binomial",
        "in_look_list",
        "find_stop",
    ]:
        match = re.search(
            r"^      (?:FUNCTION|SUBROUTINE) "
            + name
            + r"\b.*?^      END (?:FUNCTION|SUBROUTINE) "
            + name
            + r"\s*$",
            original,
            re.M | re.S,
        )
        if match is None:
            raise RuntimeError(name)
        blocks.append(match.group())
    numerical = directory / "numerical.f90"
    numerical.write_text(
        original[: original.index("    CONTAINS") + len("    CONTAINS")]
        + "\n"
        + "\n".join(blocks)
        + "\nEND MODULE\n"
    )
    driver = directory / "driver.f90"
    driver.write_text("""program reference
use design_properties_module
implicit none
integer n,side,step,i,mode
real(8) a,b,p0,alpha,chosen,achieved,lo,siglo,hi,sighi
read(*,*) n,side,step,mode,a,b,p0,alpha
if(step>1) then
 is_look_list=.true.
 length_look_list=n/step
 allocate(look_list(length_look_list))
 look_list=[(i,i=step,n,step)]
endif
call find_stop(n,a,b,p0,alpha,side,mode==1,chosen,achieved,lo,siglo,hi,sighi)
write(*,'(*(ES27.17E3,1X))') chosen,achieved,lo,siglo,hi,sighi
end program
""")
    deps = [source / (n + ".f90") for n in ["zero_finder", "cdf_aux_mod", "cdf_beta_mod"]]
    executable = directory / "reference"
    subprocess.run(
        [
            "gfortran",
            "-O2",
            "-ffree-line-length-none",
            *map(str, deps),
            str(numerical),
            str(driver),
            "-o",
            str(executable),
        ],
        cwd=directory,
        check=True,
        capture_output=True,
    )
    cases = []
    for side, alternative in [(-1, "less"), (1, "greater"), (2, "two-sided")]:
        for step in [1, 5]:
            for mode, selection in [(1, "conservative"), (2, "nearest")]:
                output = subprocess.run(
                    [str(executable)],
                    input=f"50 {side} {step} {mode} .5 .5 .2 .05\n",
                    text=True,
                    capture_output=True,
                    check=True,
                    timeout=10,
                )
                values = list(map(float, output.stdout.split()))
                if not len(values) == 6 or not 0 < values[0] < 1:
                    raise RuntimeError(f"Native calibration failed: {output.stdout}")
                cases.append(
                    dict(
                        alternative=alternative,
                        looks=list(range(step, 51, step)),
                        selection=selection,
                        tail_probability=values[0],
                        significance=values[1],
                        lower_tail=values[2],
                        lower_level=values[3],
                        upper_tail=values[4],
                        upper_level=values[5],
                    )
                )
    fixture = dict(
        source="Unchanged FIND_STOP, boundary and probability routines; independent driver",
        prior=[0.5, 0.5],
        null_probability=0.2,
        max_subjects=50,
        requested_significance=0.05,
        source_sha256=hashlib.sha256(
            (source / "design_properties_module.f90").read_bytes()
        ).hexdigest(),
        extracted_sha256=hashlib.sha256(numerical.read_bytes()).hexdigest(),
        driver_sha256=hashlib.sha256(driver.read_bytes()).hexdigest(),
        dependency_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in deps},
        compiler=subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
        cases=cases,
    )
    Path("tests/fixtures/seqbin_calibration.json").write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"Recorded {len(cases)} native calibrations")


if __name__ == "__main__":
    main()
