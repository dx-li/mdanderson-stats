"""Native SEQBIN table compaction and its subsequent null-summary mutation."""

import hashlib
import json
import re
import subprocess
from pathlib import Path


def main():
    source = Path("research/raw/SEQBIN/source/seqbin/SOURCE").resolve()
    original = (source / "design_properties_module.f90").read_text()
    directory = Path("research/raw/reference/seqbin_table").resolve()
    directory.mkdir(parents=True, exist_ok=True)
    blocks = []
    for name in [
        "bayesian_design",
        "calculate_design",
        "calculate_properties",
        "convolve_binomial",
        "in_look_list",
        "compact_table",
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
integer n,side,step,i,m,lo,hi
real(8) a,b,p0,cut,pl,ph,cl,ch,correct
read(*,*) n,side,step,a,b,p0,cut
if(step>1) then
 is_look_list=.true.
 length_look_list=n/step
 allocate(look_list(length_look_list))
 look_list=[(i,i=step,n,step)]
endif
call calculate_design(n,side,a,b,p0,cut,cut)
call calculate_properties(p0)
correct=overall_prob_quit
call compact_table(side,m)
write(*,*) m
cl=0d0
ch=0d0
do i=1,m
 lo=0
 hi=n_subject(i)
 pl=0d0
 ph=0d0
 if(side/=1) then
  lo=continue_low(i)
  pl=prob_quit_low(i)
 endif
 if(side/=-1) then
  hi=continue_high(i)
  ph=prob_quit_high(i)
 endif
 cl=cl+pl
 ch=ch+ph
 write(*,'(*(ES27.17E3,1X))') real(n_subject(i),8),real(lo,8),pl,cl,real(hi,8),ph,ch,pl+ph,cl+ch
enddo
call calculate_properties(p0)
write(*,'(*(ES27.17E3,1X))') correct,overall_prob_quit
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
            output = subprocess.run(
                [str(executable)],
                input=f"50 {side} {step} .5 .5 .2 .005\n",
                text=True,
                capture_output=True,
                check=True,
                timeout=5,
            )
            lines = output.stdout.splitlines()
            size = int(lines[0])
            correct, mutated = map(float, lines[-1].split())
            cases.append(
                dict(
                    alternative=alternative,
                    looks=list(range(step, 51, step)),
                    rows=[list(map(float, line.split())) for line in lines[1 : 1 + size]],
                    correct_significance=correct,
                    mutated_significance=mutated,
                )
            )
    fixture = dict(
        source=(
            "Unchanged COMPACT_TABLE, boundary and probability routines; independent table driver"
        ),
        max_subjects=50,
        prior=[0.5, 0.5],
        null_probability=0.2,
        tail_probability=0.005,
        source_sha256=hashlib.sha256(
            (source / "design_properties_module.f90").read_bytes()
        ).hexdigest(),
        extracted_sha256=hashlib.sha256(numerical.read_bytes()).hexdigest(),
        driver_sha256=hashlib.sha256(driver.read_bytes()).hexdigest(),
        dependency_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in deps},
        compiler=subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
        cases=cases,
    )
    Path("tests/fixtures/seqbin_table.json").write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"Recorded {len(cases)} native tables")


if __name__ == "__main__":
    main()
