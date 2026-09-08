"""Record KSBIN1 mode-3 sample-size searches using the documented reference build."""

import json
import subprocess
from pathlib import Path

from reference_binomial_power import main as build_reference


def main():
    build_reference()
    directory = Path("research/raw/reference/binomial-power").resolve()
    driver = directory / "sample_size.f90"
    driver.write_text("""program reference
use solve_binomial_one_sample_mod
use biomath_constants_mod
implicit none
integer status
real(dpkind) params(5)
read(*,*) params(1),params(2),params(4),params(5)
params(3)=2.0_dpkind
call solve_binomial_one_sample(3,params,params(1)<params(2),status)
write(*,'(I4,3(1X,ES27.17E3))') status,params(3),params(4),params(5)
end program
""")
    executable = directory / "sample_size"
    objects = [str(p) for p in directory.glob("*.o") if p.name != "ksbin1_main.o"]
    subprocess.run(
        ["gfortran", "-O2", str(driver), *objects, "-o", str(executable)],
        cwd=directory,
        check=True,
        capture_output=True,
    )
    cases = []
    for p0, pa in [(0.2, 0.06), (0.3, 0.6), (0.8, 0.94), (0.3, 0.35)]:
        for alpha in [0.01, 0.05, 0.1]:
            for target in [0.5, 0.8, 0.95]:
                result = subprocess.run(
                    [str(executable)],
                    input=f"{p0} {pa} {alpha} {target}\n",
                    text=True,
                    capture_output=True,
                    check=True,
                    timeout=10,
                )
                values = result.stdout.split()
                cases.append(
                    dict(
                        null=p0,
                        alternative=pa,
                        alpha=alpha,
                        target_power=target,
                        status=int(values[0]),
                        trials=float(values[1]),
                        significance=float(values[2]),
                        power=float(values[3]),
                    )
                )
    provenance = json.loads(Path("tests/fixtures/binomial_power.json").read_text())
    provenance.pop("cases")
    provenance["source"] = "KSBIN1 solve_binomial_one_sample mode 3; unchanged statistical routines"
    provenance["cases"] = cases
    Path("tests/fixtures/binomial_sample_size.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(f"Recorded {len(cases)} native sample-size solves")


if __name__ == "__main__":
    main()
