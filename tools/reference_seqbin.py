"""Native SEQBIN boundary and forward-probability fixtures from unchanged routines."""

import hashlib
import json
import re
import subprocess
from pathlib import Path


def main():
    source = Path("research/raw/SEQBIN/source/seqbin/SOURCE").resolve()
    original = (source / "design_properties_module.f90").read_text()
    directory = Path("research/raw/reference/seqbin").resolve()
    directory.mkdir(parents=True, exist_ok=True)
    prefix = original[: original.index("    CONTAINS") + len("    CONTAINS")]
    blocks = []
    for name in [
        "bayesian_design",
        "calculate_design",
        "calculate_properties",
        "convolve_binomial",
        "in_look_list",
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
    numerical.write_text(prefix + "\n" + "\n".join(blocks) + "\nEND MODULE\n")
    driver = directory / "driver.f90"
    driver.write_text("""program reference
use design_properties_module
implicit none
integer n,side,i,k,step
real(8) a,b,p0,cut,p
read(*,*) n,side,step,a,b,p0,cut,p
if(step>1) then
 is_look_list=.true.
 length_look_list=n/step
 allocate(look_list(length_look_list))
 look_list=[(i,i=step,n,step)]
endif
call calculate_design(n,side,a,b,p0,cut,cut)
call calculate_properties(p)
do i=1,n
 if(mod(i,step)/=0) cycle
 k=0
 if(side/=1) k=continue_low(i)
 write(*,*) k
 k=i
 if(side/=-1) k=continue_high(i)
 write(*,*) k
 if(side/=1) then
  write(*,'(ES27.17E3)') prob_quit_low(i)
 else
  write(*,*) 0d0
 endif
 if(side/=-1) then
  write(*,'(ES27.17E3)') prob_quit_high(i)
 else
  write(*,*) 0d0
 endif
enddo
write(*,'(*(ES27.17E3,1X))') overall_prob_quit,expected_n, &
 expected_n_quit_low,expected_n_quit_high
end program
""")
    executable = directory / "reference"
    dependencies = [source / (n + ".f90") for n in ["zero_finder", "cdf_aux_mod", "cdf_beta_mod"]]
    subprocess.run(
        [
            "gfortran",
            "-O2",
            "-ffree-line-length-none",
            *map(str, dependencies),
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
    for prior in [[1, 1], [2, 8], [30, 1]]:
        for side, alternative in [(-1, "less"), (1, "greater"), (2, "two-sided")]:
            for step in [1, 5]:
                output = subprocess.run(
                    [str(executable)],
                    input=f"20 {side} {step} {prior[0]} {prior[1]} .2 .05 .2\n",
                    text=True,
                    capture_output=True,
                    check=True,
                    timeout=5,
                )
                values = list(map(float, output.stdout.split()))
                rows = [values[i : i + 4] for i in range(0, len(values) - 4, 4)]
                cases.append(
                    dict(
                        prior=prior,
                        alternative=alternative,
                        looks=list(range(step, 21, step)),
                        continue_low=[int(r[0]) for r in rows],
                        continue_high=[int(r[1]) for r in rows],
                        quit_low=[r[2] for r in rows],
                        quit_high=[r[3] for r in rows],
                        rejection=values[-4],
                        expected_subjects=values[-3],
                        expected_low=values[-2],
                        expected_high=values[-1],
                    )
                )
    files = [source / "design_properties_module.f90", *dependencies]
    fixture = dict(
        source=(
            "Unchanged SEQBIN boundary/forward-probability routines and beta dependencies; "
            "independent driver"
        ),
        hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        extracted_sha256=hashlib.sha256(numerical.read_bytes()).hexdigest(),
        driver_sha256=hashlib.sha256(driver.read_bytes()).hexdigest(),
        compiler=subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
        cases=cases,
    )
    Path("tests/fixtures/seqbin.json").write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"Recorded {len(cases)} native cases")


if __name__ == "__main__":
    main()
