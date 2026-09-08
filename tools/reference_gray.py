"""Unchanged CUMINC CRSTM/CRST native score and covariance reference."""

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np


def main():
    source = Path("research/raw/CUMINC/source/S/cuminc.f")
    directory = Path("research/raw/reference/gray").resolve()
    directory.mkdir(parents=True, exist_ok=True)
    numerical = directory / "gray.f"
    numerical.write_text(
        "      SUBROUTINE crst(" + source.read_text().split("      SUBROUTINE crst(", 1)[1]
    )
    driver = directory / "driver.f90"
    driver.write_text("""program reference
implicit none
integer n,ng,nst,i,j,nv
real(8) rho
real(8),allocatable::y(:),s(:),vs(:,:),ys(:),v(:),st(:),vt(:),wk(:)
integer,allocatable::m(:),ig(:),ist(:),ms(:),igs(:),iwk(:)
read(*,*) n,ng,nst,rho
nv=ng*(ng-1)/2
allocate(y(n),m(n),ig(n),ist(n),s(ng-1),vs(ng-1,ng-1),ys(n),ms(n),igs(n))
allocate(v(nv),st(ng-1),vt(nv),wk(ng*(4+3*ng)),iwk(4*ng))
do i=1,n
 read(*,*)y(i),m(i),ig(i),ist(i)
enddo
call crstm(y,m,ig,ist,n,rho,nst,ng,s,vs,ys,ms,igs,v,st,vt,wk,iwk)
write(*,'(*(ES27.17E3,1X))')s
do i=1,ng-1
 write(*,'(*(ES27.17E3,1X))')(vs(i,j),j=1,ng-1)
enddo
end program
""")
    executable = directory / "reference"
    subprocess.run(
        ["gfortran", "-O2", "-std=legacy", str(numerical), str(driver), "-o", str(executable)],
        cwd=directory,
        check=True,
        capture_output=True,
    )
    rng = np.random.default_rng(90184)
    inputs = []
    for ng in [2, 3, 4]:
        for ties in [False, True]:
            for stratified in [False, True]:
                n = 80
                times = rng.uniform(0, 12, n)
                if ties:
                    times = np.floor(times)
                group = np.tile(np.arange(1, ng + 1), n // ng + 1)[:n]
                strata = rng.integers(1, 4, n) if stratified else np.ones(n, dtype=int)
                event = rng.integers(0, 3, n)
                for rho in [-0.5, 0, 1, 2]:
                    inputs.append((times, event, group, strata, rho))
    data_path = Path("research/raw/CUMINC/source/S/test.data")
    data = np.loadtxt(data_path)
    for cause in [1, 2]:
        for stratified in [False, True]:
            status = np.where(data[:, 1] == 0, 0, np.where(data[:, 1] == cause, 1, 2))
            inputs.append(
                (
                    data[:, 0],
                    status,
                    data[:, 2],
                    data[:, 3] if stratified else np.ones(len(data)),
                    0,
                )
            )
    for event in [[0, 0, 0, 0], [2, 2, 2, 2], [1, 1, 1, 1], [1, 2, 1, 0]]:
        inputs.append(
            (np.array([1, 1, 2, 2]), np.array(event), np.array([1, 2, 1, 2]), np.ones(4), 0)
        )
    # Strata need not contain every group; retain the global contrast basis.
    inputs.append(
        (
            np.arange(9),
            np.array([1, 2, 0] * 3),
            np.array([1, 1, 2, 2, 2, 3, 1, 3, 3]),
            np.array([1, 1, 1, 2, 2, 2, 3, 3, 3]),
            0,
        )
    )
    cases = []
    for times, event, group, strata, rho in inputs:
        order = np.argsort(times, kind="stable")
        groups, group_codes = np.unique(group, return_inverse=True)
        levels, stratum_codes = np.unique(strata, return_inverse=True)
        ng, nst = len(groups), len(levels)
        text = (
            f"{len(times)} {ng} {nst} {rho}\n"
            + "\n".join(
                f"{times[i]:.17g} {int(event[i])} "
                f"{int(group_codes[i]) + 1} {int(stratum_codes[i]) + 1}"
                for i in order
            )
            + "\n"
        )
        output = subprocess.run(
            [str(executable)], input=text, text=True, capture_output=True, check=True, timeout=10
        )
        rows = [list(map(float, row.split())) for row in output.stdout.splitlines()]
        if not all(np.all(np.isfinite(row)) for row in rows):
            raise RuntimeError(f"Native nonfinite case {len(cases)}: {text}")
        cases.append(
            dict(
                time=times.tolist(),
                event=event.tolist(),
                group=group.tolist(),
                strata=strata.tolist(),
                rho=rho,
                score=rows[0],
                covariance=rows[1:],
            )
        )
    fixture = dict(
        source="Unchanged CUMINC CRSTM/CRST with independent driver",
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        extracted_sha256=hashlib.sha256(numerical.read_bytes()).hexdigest(),
        driver_sha256=hashlib.sha256(driver.read_bytes()).hexdigest(),
        data_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
        compiler=subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
        cases=cases,
    )
    Path("tests/fixtures/gray.json").write_text(
        json.dumps(fixture, indent=2, allow_nan=False) + "\n"
    )
    print(f"Recorded {len(cases)} native Gray tests")


if __name__ == "__main__":
    main()
