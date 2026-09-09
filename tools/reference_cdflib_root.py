"""Audit original direct and reverse-communication root-finder contracts."""

import hashlib
import json
import subprocess
import tarfile
import tempfile
from pathlib import Path

from audit_cdflib90 import ARCHIVE_SHA256, F95_ROOT

DRIVER = """program reference
use biomath_constants_mod
use zero_finder
implicit none
type(zf_locals) local
integer mode,function_id,status,final_status,calls,steps
real(dpkind) target,answer,fx,low,high,initial,atol,rtol,left_end,right_end
logical crash_left,crash_hi
read(*,*) mode,function_id,low,high,target,initial,atol,rtol
call set_zero_finder(low_limit=low,hi_limit=high,abs_tol=atol, &
 rel_tol=rtol,local=local)
calls=0
answer=initial
fx=0.0_dpkind
select case(mode)
case(1)
call interval_zf(evaluate,target,answer,status,local)
case(2,4)
status=0
do steps=1,10000
if(mode==2) then
call rc_interval_zf(status,answer,fx,local)
else
call rc_step_zf(status,answer,fx,local)
end if
if(status/=1) exit
fx=evaluate(answer)-target
end do
if(status==1) stop 2
case(3)
call step_zf(evaluate,target,answer,status,local)
end select
call final_zf_state(local,final_status,crash_left,crash_hi,left_end,right_end)
write(*,'(A,4(I0,1X))') 'S ',status,final_status,zf_current_status,calls
if(status==0.or.mode/=1) write(*,'(A,ES26.17E3)') 'A ',answer
if(status==-1) write(*,'(A,2(L1,1X),2(ES26.17E3,1X))') &
 'F ',crash_left,crash_hi,left_end,right_end
contains
function evaluate(x) result(y)
real(dpkind),intent(in)::x
real(dpkind)::y
calls=calls+1
write(*,'(A,ES26.17E3)') 'X ',x
select case(function_id)
case(1)
y=x
case(2)
y=x*x
case(3)
y=-x
end select
end function evaluate
end program reference
"""


def cases():
    for mode in range(1, 5):
        for function_id, targets in [
            (1, [0, 0.5, 1, 2, 3]),
            (2, [-1, 0, 2, 4, 9]),
            (3, [-1, 1, -3]),
        ]:
            for target in targets:
                yield dict(
                    mode=mode,
                    function_id=function_id,
                    low=0.0,
                    high=2.0,
                    target=float(target),
                    initial=1.0,
                    abs_tol=1e-12,
                    rel_tol=1e-12,
                )
        yield dict(
            mode=mode,
            function_id=2,
            low=0.0,
            high=2.0,
            target=2.0,
            initial=1.0,
            abs_tol=0.01,
            rel_tol=0.01,
        )
        yield dict(
            mode=mode,
            function_id=2,
            low=-2.0,
            high=2.0,
            target=1.0,
            initial=0.5,
            abs_tol=1e-12,
            rel_tol=1e-12,
        )


def parse(output):
    trace = []
    result = dict(answer=None, failure=None)
    for line in output.splitlines():
        key, *fields = line.split()
        if key == "X":
            trace.append(float(fields[0]))
        elif key == "S":
            result.update(
                zip(
                    ("status", "final_status", "global_status", "evaluations"),
                    map(int, fields),
                    strict=True,
                )
            )
        elif key == "A":
            result["answer"] = float(fields[0])
        elif key == "F":
            result["failure"] = dict(
                crash_left=fields[0] == "T",
                crash_hi=fields[1] == "T",
                low=float(fields[2]),
                high=float(fields[3]),
            )
        else:
            raise RuntimeError(f"unexpected native root record: {line}")
    result["requests"] = trace
    if result.get("evaluations") != len(trace):
        raise RuntimeError("root evaluation count does not match the recorded requests")
    return result


def main():
    archive = Path("research/raw/CDFLIB90/CDFLIB90  _V90.tar.gz")
    if hashlib.sha256(archive.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        raise RuntimeError("CDFLIB90 archive hash mismatch")
    sources = {}
    with tarfile.open(archive) as tar:
        for name in ("biomath_constants_mod", "zero_finder"):
            member = tar.extractfile(F95_ROOT + "source/" + name + ".f90")
            if member is None:
                raise RuntimeError(f"missing {name} source")
            sources[name] = member.read()
    flags = ["-O0", "-ffp-contract=off", "-fcheck=all"]
    records = []
    with tempfile.TemporaryDirectory(prefix="cdflib-root-") as directory:
        work = Path(directory)
        for name, raw in sources.items():
            (work / (name + ".f90")).write_bytes(raw)
        (work / "driver.f90").write_text(DRIVER)
        subprocess.run(
            [
                "gfortran",
                *flags,
                *(name + ".f90" for name in sources),
                "driver.f90",
                "-o",
                "reference",
            ],
            cwd=work,
            check=True,
            timeout=60,
        )
        for case in cases():
            text = " ".join(str(value) for value in case.values()) + "\n"
            try:
                completed = subprocess.run(
                    [str(work / "reference")],
                    input=text,
                    text=True,
                    capture_output=True,
                    timeout=3,
                    check=True,
                )
            except subprocess.TimeoutExpired:
                records.append(dict(**case, execution_outcome="timeout"))
            else:
                records.append(
                    dict(**case, execution_outcome="completed", **parse(completed.stdout))
                )
    result = dict(
        archive_sha256=ARCHIVE_SHA256,
        source_hashes={name: hashlib.sha256(raw).hexdigest() for name, raw in sources.items()},
        driver_sha256=hashlib.sha256(DRIVER.encode()).hexdigest(),
        flags=flags,
        compiler=subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
        modes={1: "interval_zf", 2: "rc_interval_zf", 3: "step_zf", 4: "rc_step_zf"},
        functions={1: "x", 2: "x*x", 3: "-x"},
        cases=records,
    )
    Path("tests/fixtures/cdflib_root.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Recorded {len(records)} native root calls")


if __name__ == "__main__":
    main()
