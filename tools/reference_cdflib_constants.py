"""Record all constants and floating-point models from the pinned F95 source."""

import hashlib
import json
import re
import subprocess
import tarfile
import tempfile
from pathlib import Path

from audit_cdflib90 import ARCHIVE_SHA256, F95_ROOT


def main():
    archive = Path("research/raw/CDFLIB90/CDFLIB90  _V90.tar.gz")
    if hashlib.sha256(archive.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        raise RuntimeError("CDFLIB90 archive hash mismatch")
    member = F95_ROOT + "source/biomath_constants_mod.f90"
    with tarfile.open(archive) as tar:
        extracted = tar.extractfile(member)
        if extracted is None:
            raise RuntimeError("missing constants source")
        raw = extracted.read()
    source = raw.decode()
    integers = [
        declaration.split("=", 1)[0].strip()
        for statement in re.findall(r"^\s*INTEGER, PARAMETER :: (.+)$", source, re.M)
        for declaration in statement.split(",")
    ]
    reals = re.findall(r"REAL \(dpkind\), PARAMETER :: (\w+)", source)
    driver = ["program reference", "use biomath_constants_mod", "implicit none"]
    driver += [f"write(*,'(I0)') {name}" for name in integers]
    driver += [f"write(*,'(ES26.17E3)') {name}" for name in reals]
    for kind in ("dpkind", "spkind"):
        value = f"0.0_{kind}"
        for intrinsic in ("storage_size", "radix", "digits", "minexponent", "maxexponent"):
            driver.append(f"write(*,'(I0)') {intrinsic}({value})")
        for intrinsic in ("epsilon", "tiny", "huge"):
            driver.append(f"write(*,'(ES26.17E3)') {intrinsic}({value})")
    driver.append("end program reference")
    code = "\n".join(driver) + "\n"
    flags = ["-O0", "-ffp-contract=off", "-fcheck=all"]
    with tempfile.TemporaryDirectory(prefix="cdflib-constants-") as directory:
        work = Path(directory)
        (work / "constants.f90").write_bytes(raw)
        (work / "driver.f90").write_text(code)
        subprocess.run(
            ["gfortran", *flags, "constants.f90", "driver.f90", "-o", "reference"],
            cwd=work,
            check=True,
            timeout=60,
        )
        output = iter(
            subprocess.check_output([str(work / "reference")], text=True, timeout=5).splitlines()
        )
    records = {name: int(next(output)) for name in integers}
    records.update({name: float(next(output)) for name in reals})
    models = {}
    for kind in ("dpkind", "spkind"):
        model = {
            key: int(next(output))
            for key in ("bits", "radix", "digits", "minexponent", "maxexponent")
        }
        model.update({key: float(next(output)) for key in ("epsilon", "tiny", "huge")})
        models[kind] = model
    if list(output):
        raise RuntimeError("unexpected constants driver output")
    result = dict(
        archive_sha256=ARCHIVE_SHA256,
        source_member=member,
        source_sha256=hashlib.sha256(raw).hexdigest(),
        driver_sha256=hashlib.sha256(code.encode()).hexdigest(),
        compiler=subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
        flags=flags,
        integer_names=integers,
        real_names=reals,
        values=records,
        models=models,
    )
    Path("tests/fixtures/cdflib_constants.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
