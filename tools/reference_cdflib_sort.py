"""Record unchanged F95 sorting overloads, comparator contracts and failures."""

import hashlib
import json
import subprocess
from pathlib import Path

DRIVER = """program reference
use biomath_sort_mod
implicit none
integer kind,n,ncol,descending,width,i,j
real(8),allocatable :: d(:)
real(4),allocatable :: s(:)
integer,allocatable :: k(:)
character(:),allocatable :: c(:)
read(*,*) kind,n,ncol,descending,width
select case(kind)
case(1)
 allocate(d(n))
 if(n>0) read(*,*) d
 if(descending==0) then
  call sort_list(d,ncol)
 else
  call sort_list(d,ncol,dcmp)
 end if
 do i=1,n
  write(*,'(ES26.17E3)') d(i)
 end do
case(2)
 allocate(s(n))
 if(n>0) read(*,*) s
 if(descending==0) then
  call sort_list(s,ncol)
 else
  call sort_list(s,ncol,scmp)
 end if
 do i=1,n
  write(*,'(ES26.17E3)') s(i)
 end do
case(3)
 allocate(k(n))
 if(n>0) read(*,*) k
 if(descending==0) then
  call sort_list(k,ncol)
 else
  call sort_list(k,ncol,kcmp)
 end if
 do i=1,n
  write(*,'(I0)') k(i)
 end do
case(4)
 allocate(character(width)::c(n))
 do i=1,n
  read(*,'(A)') c(i)
 end do
 if(descending==0) then
  call sort_list(c,ncol)
 else
  call sort_list(c,ncol,ccmp)
 end if
 do i=1,n
  write(*,'(*(I0,1X))') (iachar(c(i)(j:j)),j=1,width)
 end do
end select
contains
logical function dcmp(a,b)
real(8) a,b
dcmp=a<b
end function
logical function scmp(a,b)
real(4) a,b
scmp=a<b
end function
logical function kcmp(a,b)
integer a,b
kcmp=a<b
end function
logical function ccmp(a,b)
character(*) a,b
ccmp=a<b
end function
end program reference
"""


def main():
    source = Path("research/raw/CDFLIB90/source/CDFLIB90/source/cdflib90_1.2/source").resolve()
    paths = [source / f"{name}.f90" for name in ("biomath_constants_mod", "biomath_sort_mod")]
    work = Path("research/raw/reference/cdflib-sort").resolve()
    work.mkdir(parents=True, exist_ok=True)
    driver = work / "driver.f90"
    driver.write_text(DRIVER)
    exe = work / "reference"
    command = [
        "gfortran",
        "-O0",
        "-ffp-contract=off",
        "-fcheck=all",
        *map(str, paths),
        str(driver),
        "-o",
        str(exe),
    ]
    subprocess.run(command, cwd=work, check=True)
    cases = []

    def record(kind, values, ncol, descending, label):
        width = max([1, *[len(v) for v in values]]) if kind == 4 else 1
        payload = f"{kind} {len(values)} {ncol} {descending} {width}\n"
        payload += "\n".join(values) + "\n" if kind == 4 else " ".join(map(str, values)) + "\n"
        case = dict(
            kind=kind,
            values=values,
            ncol=ncol,
            descending=bool(descending),
            label=label,
            width=width,
        )
        try:
            result = subprocess.run(
                [str(exe)], input=payload, text=True, capture_output=True, timeout=3, check=True
            )
        except subprocess.TimeoutExpired:
            case.update(execution_outcome="timeout", timeout_seconds=3)
        except subprocess.CalledProcessError as error:
            case.update(
                execution_outcome="process_error", exit_code=error.returncode, stderr=error.stderr
            )
        else:
            lines = result.stdout.splitlines()
            if len(lines) != len(values):
                raise RuntimeError("unexpected native sorting output")
            output = (
                ["".join(chr(int(v)) for v in line.split()) for line in lines]
                if kind == 4
                else [float(line) if kind in (1, 2) else int(line) for line in lines]
            )
            case.update(execution_outcome="completed", result=output)
        cases.append(case)

    numeric = [
        [],
        [1],
        [3, -2, 1, 0, -1],
        list(range(15)),
        list(range(14, -1, -1)),
        [2] * 15,
        [i % 3 for i in range(24)],
    ]
    for kind in (1, 2, 3):
        for index, values in enumerate(numeric):
            for descending in (0, 1):
                record(kind, values, len(values), descending, f"ordinary-{index}")
        record(kind, [3, 1, 2, 99, -7], 3, 0, "prefix")
    for kind in (1, 2):
        values = [0.25, -2.75, 1e-40, -1e-40, 1e20, 1.17549435e-38, 0.0, -0.0]
        for descending in (0, 1):
            record(kind, values, len(values), descending, "fractional-and-tiny")
    for descending in (0, 1):
        record(1, [1e300, -1e300, 1e-300, -1e-300], 4, descending, "wide-double")
    for values, label in [
        ([], "empty"),
        (["B", "a", "A", "a "], "letters"),
        (["a\t", "a", "a!", " A"], "blank-padding"),
        (["same"] * 15, "ties"),
        ([f"{i:02d}" for i in range(14, -1, -1)], "partition"),
        (["b" * 300, "a" * 300, "c" * 300], "long-strings"),
    ]:
        for descending in (0, 1):
            record(4, values, len(values), descending, label)
    record(4, ["c", "a", "b", "z"], 3, 0, "prefix")
    archive = Path("research/raw/CDFLIB90/CDFLIB90  _V90.tar.gz")
    data = dict(
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        command=command,
        compiler=subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
        driver=DRIVER,
        adaptations=[],
        cases=cases,
    )
    Path("tests/fixtures/cdflib_sort.json").write_text(json.dumps(data, indent=2) + "\n")
    print([(c["kind"], c["label"], c["descending"], c["execution_outcome"]) for c in cases])


if __name__ == "__main__":
    main()
