"""Probe SPPCR RNG streams, state operations and single-precision binomials."""

import hashlib
import json
import subprocess
import tempfile
import zipfile
from pathlib import Path

from reference_sppcr import ARCHIVE, SHA256

DRIVER = """program probe
use ecuyer_cote_mod
use random_standard_uniform_mod
use random_binomial_mod
implicit none
integer :: a,b,g,i,k,n,count,ios
real :: p
character(256) :: line
character(20) :: op
character(80) :: phrase
call set_all_seeds(1234567890,123456789)
do
 read(*,'(A)',iostat=ios) line
 if (ios/=0) exit
 read(line,*) op
 select case(trim(op))
 case('select')
  read(line,*) op,g
  call set_current_generator(g)
 case('all')
  read(line,*) op,a,b
  call set_all_seeds(a,b)
 case('seed')
  read(line,*) op,a,b
  call set_current_seed(a,b)
 case('anti')
  read(line,*) op,k
  call set_antithetic(k==1)
 case('reset')
  read(line,*) op,k
  call reinitialize_current_generator(k)
 case('advance')
  read(line,*) op,k
  call advance_state(k)
 case('phrase')
  read(*,'(A)') phrase
  call phrase_to_seed(phrase,a,b)
  write(*,'(A,2I14)') 'phrase ',a,b
  call set_all_seeds(a,b)
 case('integer')
  read(line,*) op,count
  write(*,'(A,1000I14)') 'integer ',(random_large_integer(),i=1,count)
 case('uniform')
  read(line,*) op,count
  write(*,'(A,1000ES25.16E3)') 'uniform ',(random_standard_uniform(),i=1,count)
 case('binomial')
  read(line,*) op,n,p,count
  write(*,'(A,1000I14)') 'binomial ',(random_binomial(n,p),i=1,count)
 end select
 call get_current_generator(g)
 call get_seeds(a,b)
 write(*,'(A,3I14)') 'state ',g,a,b
end do
end program
"""


def main():
    if hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() != SHA256:
        raise RuntimeError("SPPCR archive changed")
    names = [
        "ecuyer_cote_mod.f90",
        "random_standard_uniform_mod.f90",
        "random_binomial_mod.f90",
        "phrase_to_seed.f90",
    ]
    with zipfile.ZipFile(ARCHIVE) as z:
        contents = {n: z.read("sppcr/source/" + n) for n in names}
    cases = {f"stream_{g}": [f"select {g}", "integer 16", "uniform 16"] for g in range(1, 33)}
    cases.update(
        {
            f"anti_{g}": [
                f"select {g}",
                "anti 1",
                "integer 16",
                "uniform 16",
                "anti 0",
                "integer 8",
            ]
            for g in [1, 2, 7, 32]
        }
    )
    cases["blocks"] = [
        "integer 9",
        "reset 0",
        "integer 9",
        "reset 1",
        "integer 9",
        "reset -1",
        "integer 9",
    ]
    cases["selected_seed"] = [
        "select 7",
        "seed 111 222",
        "integer 20",
        "reset -1",
        "integer 20",
        "select 1",
        "integer 10",
    ]
    cases["reseed_stream_1"] = ["integer 5", "all 111 222", "integer 10", "select 32", "integer 10"]
    cases["reseed_other_stream_defect"] = [
        "integer 5",
        "select 7",
        "all 111 222",
        "select 1",
        "integer 10",
    ]
    for k in [0, 1, 10, 30, 64]:
        cases[f"advance_{k}"] = [
            "integer 3",
            f"advance {k}",
            "integer 20",
            "reset -1",
            "integer 20",
        ]
    for i, phrase in enumerate(
        ["", "SPPCR seed", "SPPCR seed   ", "!@#$%^&*()_+[];:'\"<>?,./", "123456.789", "a" * 80]
    ):
        cases[f"phrase_{i}"] = ["phrase", phrase, "integer 20"]
    for n in [0, 1, 20, 100, 1000, 1000000]:
        for p in [0, 1, 0.0001, 0.1, 0.3, 0.5, 0.8, 0.9999]:
            cases[f"binomial_{n}_{p}"] = [f"binomial {n} {p} 100", "integer 4"]
    cases["changing_parameters"] = [
        "binomial 100 .3 20",
        "binomial 100 .3 20",
        "binomial 200 .8 20",
        "binomial 10 .1 20",
        "binomial 100 .3 20",
    ]
    cases["generation_grid"] = [
        f"binomial {n} {p} 1"
        for _ in range(3)
        for n, probabilities in [(20, [0.1, 0.4, 0.8]), (40, [0.2, 0.5, 0.9])]
        for p in probabilities
    ]
    records = {}
    with tempfile.TemporaryDirectory(prefix="sppcr-random-") as tmp:
        w = Path(tmp)
        for n, content in contents.items():
            (w / n).write_bytes(content)
        (w / "probe.f90").write_text(DRIVER)
        flags = ["gfortran", "-std=legacy", "-O0", "-ffp-contract=off", "-fcheck=all"]
        subprocess.run(
            [*flags, *names, "probe.f90", "-o", "probe"],
            cwd=w,
            check=True,
            capture_output=True,
            timeout=60,
        )
        for name, commands in cases.items():
            r = subprocess.run(
                [str(w / "probe")],
                input="\n".join(commands) + "\n",
                text=True,
                capture_output=True,
                timeout=3,
                check=True,
            )
            records[name] = dict(commands=commands, stdout=r.stdout)
    report = dict(
        archive_sha256=SHA256,
        sources={n: hashlib.sha256(b).hexdigest() for n, b in contents.items()},
        driver=DRIVER,
        compiler=subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
        cases=records,
    )
    Path("tests/fixtures/sppcr_random.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
