"""Record native FGH objectives/derivatives and the opposite-shape chain rule."""

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np


def main():
    source = Path("research/raw/STUKEL/S/stukel/all.f")
    directory = Path("research/raw/reference/stukel")
    directory.mkdir(parents=True, exist_ok=True)
    driver = directory / "objective.f90"
    driver.write_text("""program reference
implicit none
integer n,p,k,family,i,lh
logical bad
double precision a(2),f
double precision,allocatable :: x(:,:),y(:),m(:),h(:),b(:),g(:),hess(:)
read(*,*) n,p,k,family,a
lh=k*(k+1)/2
allocate(x(n,p),y(n),m(n),h(n),b(k),g(k),hess(lh))
read(*,*) x
read(*,*) y
read(*,*) m
read(*,*) b
call fgh(2,family,n,p,k,x,y,m,b,a,lh,bad,h,f,g,hess)
if (bad) stop 1
write(*,'(es27.17e3)') f
write(*,'(*(es27.17e3,1x))') g
write(*,'(*(es27.17e3,1x))') hess
end program
""")
    executable = directory / "objective"
    subprocess.run(
        ["gfortran", "-O2", "-std=legacy", str(source), str(driver), "-o", str(executable)],
        check=True,
        capture_output=True,
    )
    x = np.column_stack((np.ones(6), [-2, -1, -0.2, 0.4, 1, 2]))
    y, n, fixed = [0, 2, 4, 8, 10, 9], [10] * 6, [-0.3, 0.4]

    def evaluate(family, coef):
        data = f"6 2 {len(coef)} {family} {fixed[0]} {fixed[1]}\n"
        for values in (x.ravel(order="F"), y, n, coef):
            data += " ".join(map(str, values)) + "\n"
        output = subprocess.run(
            [str(executable)], input=data, text=True, check=True, capture_output=True
        ).stdout.splitlines()
        hessian = np.zeros((len(coef), len(coef)))
        hessian[np.tril_indices(len(coef))] = [float(v) for v in output[2].split()]
        hessian += np.tril(hessian, -1).T
        return {
            "objective": float(output[0]),
            "gradient": [float(v) for v in output[1].split()],
            "hessian": hessian.tolist(),
        }

    cases = []
    for family in range(6):
        for shape in (-0.3, 0, 0.2, 0.001):
            coef = [0.2, 0.7] + ([shape, -0.1] if family == 5 else [shape] if family else [])
            case = {
                "x": x.tolist(),
                "successes": y,
                "trials": n,
                "family": family,
                "fixed_alpha": fixed,
                "coefficients": coef,
                "reference": evaluate(family, coef),
            }
            if family == 4:
                full = evaluate(5, [*coef[:2], shape, -shape])
                transform = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1]])
                case["chain_rule_reference"] = {
                    "objective": full["objective"],
                    "gradient": (transform.T @ full["gradient"]).tolist(),
                    "hessian": (transform.T @ full["hessian"] @ transform).tolist(),
                }
            cases.append(case)
    result = {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "compiler": subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
        "notes": "Original FGH icode=2. Family 4 also records J-transformed family-5 "
        "derivatives to expose its omitted alpha2=-alpha1 chain rule.",
        "cases": cases,
    }
    Path("tests/fixtures/stukel_objective.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(f"Recorded {len(cases)} original FGH derivative cases")


if __name__ == "__main__":
    main()
