"""Record KSBIN1's alternative-probability solve mode using the documented reference build."""

import json
import subprocess
from pathlib import Path

from reference_binomial_power import main as build_reference


def main():
    build_reference()
    directory = Path("research/raw/reference/binomial-power").resolve()
    driver = directory / "alternative.f90"
    driver.write_text("""program reference
use solve_binomial_one_sample_mod
use biomath_constants_mod
implicit none
integer status,direction
real(dpkind) params(5), c1,s1,w1,c2,s2,w2
read(*,*) params(1),direction,params(3),params(5)
params(2)=0.5_dpkind
params(4)=0.05_dpkind
call solve_binomial_one_sample(2,params,direction==1,status)
write(*,'(I4,3(1X,ES27.17E3))') status,params(2),params(4),params(5)
end program
""")
    executable = directory / "alternative"
    objects = [str(p) for p in directory.glob("*.o") if p.name != "ksbin1_main.o"]
    subprocess.run(
        ["gfortran", "-O2", str(driver), *objects, "-o", str(executable)],
        cwd=directory,
        check=True,
        capture_output=True,
    )
    cases = []
    for n in [30, 50, 100]:
        for p0, direction in [(0.2, 0), (0.3, 1), (0.8, 1)]:
            for target in [0.5, 0.8, 0.95]:
                result = subprocess.run(
                    [str(executable)],
                    input=f"{p0} {direction} {n} {target}\n",
                    text=True,
                    capture_output=True,
                    check=True,
                    timeout=10,
                )
                values = result.stdout.split()
                cases.append(
                    dict(
                        trials=n,
                        null=p0,
                        direction="greater" if direction else "less",
                        alternative=float(values[1]),
                        alpha=0.05,
                        target_power=target,
                        status=int(values[0]),
                        significance=float(values[2]),
                        power=float(values[3]),
                    )
                )
    provenance = json.loads(Path("tests/fixtures/binomial_power.json").read_text())
    provenance.pop("cases")
    provenance["source"] = "KSBIN1 solve_binomial_one_sample mode 2; unchanged statistical routines"
    provenance["cases"] = cases
    Path("tests/fixtures/binomial_alternative.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(f"Recorded {len(cases)} native alternative-probability solves")


if __name__ == "__main__":
    main()
