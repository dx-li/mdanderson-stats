"""Record KSBIN1's significance solve mode using the documented reference build."""

import json
import subprocess
from pathlib import Path

from reference_binomial_power import main as build_reference


def main():
    build_reference()
    directory = Path("research/raw/reference/binomial-power").resolve()
    driver = directory / "significance.f90"
    driver.write_text("""program reference
use solve_binomial_one_sample_mod
use biomath_constants_mod
implicit none
integer status
real(dpkind) params(5), c1,s1,w1,c2,s2,w2
read(*,*) params(1),params(2),params(3),params(5)
params(4)=0.05_dpkind
call solve_binomial_one_sample(4,params,params(1)<params(2),status)
write(*,'(I4,2(1X,ES27.17E3))') status,params(4),params(5)
end program
""")
    executable = directory / "significance"
    objects = [str(p) for p in directory.glob("*.o") if p.name != "ksbin1_main.o"]
    subprocess.run(
        ["gfortran", "-O2", str(driver), *objects, "-o", str(executable)],
        cwd=directory,
        check=True,
        capture_output=True,
    )
    cases = []
    for n in [20, 50, 100]:
        for p0, pa in [(0.2, 0.06), (0.3, 0.6), (0.8, 0.94)]:
            for target in [0.5, 0.8, 0.95]:
                result = subprocess.run(
                    [str(executable)],
                    input=f"{p0} {pa} {n} {target}\n",
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
                        alternative=pa,
                        target_power=target,
                        status=int(values[0]),
                        significance=float(values[1]),
                        power=float(values[2]),
                    )
                )
    provenance = json.loads(Path("tests/fixtures/binomial_power.json").read_text())
    provenance.pop("cases")
    provenance["source"] = "KSBIN1 solve_binomial_one_sample mode 4; unchanged statistical routines"
    provenance["cases"] = cases
    Path("tests/fixtures/binomial_significance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(f"Recorded {len(cases)} native significance solves")


if __name__ == "__main__":
    main()
