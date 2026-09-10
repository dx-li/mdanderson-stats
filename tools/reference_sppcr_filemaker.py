"""Native SPPCR FileMaker-style numeric-row input probes."""

import hashlib
import json
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path

from reference_sppcr import ARCHIVE, SHA256
from reference_sppcr_batch import DRIVER as BATCH_DRIVER

DRIVER = BATCH_DRIVER.replace("read_one_bat", "read_mjs_one")
BASE = "100 102 20 .5 100 4 102 8 104 2\n100 102 30 1 100 10 102 15 104 5\n"


def main():
    if hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() != SHA256:
        raise RuntimeError("SPPCR archive changed")
    with zipfile.ZipFile(ARCHIVE) as z:
        contents = {i.filename: z.read(i) for i in z.infolist() if not i.is_dir()}
    prefix = "sppcr/source/"
    order = re.findall(r"-c (\w+\.f90)", contents[prefix + "compile.sppcr"].decode())
    cases = {
        "basic": BASE,
        "quoted_csv": "\n".join(
            ",".join('"' + x + '"' for x in row.split()) for row in BASE.splitlines()
        )
        + "\n",
        "exponent_comments": BASE.replace(".5", "5d-1")
        .replace("104 2\n", "104 2 # first\n")
        .replace("104 5\n", "104 5 / second\n"),
        "homozygote": BASE.replace("100 102 20", "100 100 20").replace("100 102 30", "100 100 30"),
        "drop_unseen": BASE.replace("104 2", "104 0").replace("104 5", "104 0"),
    }
    records = {}
    with tempfile.TemporaryDirectory(prefix="sppcr-filemaker-") as tmp:
        w = Path(tmp)
        for n in order:
            (w / n).write_bytes(contents[prefix + n])
        (w / "probe.f90").write_text(DRIVER)
        flags = ["gfortran", "-std=legacy", "-O0", "-ffp-contract=off", "-fcheck=all"]
        subprocess.run([*flags, "-c", *order], cwd=w, check=True, capture_output=True, timeout=60)
        subprocess.run(
            [
                *flags,
                "probe.f90",
                *[str(Path(n).with_suffix(".o")) for n in order if n != "sppcr.f90"],
                "-o",
                "probe",
            ],
            cwd=w,
            check=True,
            capture_output=True,
            timeout=60,
        )
        for name, value in cases.items():
            r = subprocess.run(
                [str(w / "probe")], input="0\n" + value, text=True, capture_output=True, timeout=3
            )
            records[name] = dict(
                input=value, returncode=r.returncode, stdout=r.stdout, stderr=r.stderr
            )
    report = dict(
        archive_sha256=SHA256,
        driver=DRIVER,
        scope=(
            "Original FileMaker numeric-row reader and conversion, with unused input arrays zeroed"
        ),
        cases=records,
    )
    Path("tests/fixtures/sppcr_filemaker.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
