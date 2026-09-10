"""Native SPPCR interactive input probes."""

import hashlib
import json
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path

from reference_sppcr import ARCHIVE, SHA256
from reference_sppcr_batch import DRIVER as BATCH_DRIVER

DRIVER = BATCH_DRIVER.replace("read_one_bat", "interactive_read").replace(
    "call interactive_read(5)", "call interactive_read"
)
BASE = "2\n3\n100 102 104\n100 102\n.5 1\n20 30\n4 8 2\n10 15 5\n"


def main():
    if hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() != SHA256:
        raise RuntimeError("SPPCR archive changed")
    with zipfile.ZipFile(ARCHIVE) as z:
        contents = {i.filename: z.read(i) for i in z.infolist() if not i.is_dir()}
    prefix = "sppcr/source/"
    order = re.findall(r"-c (\w+\.f90)", contents[prefix + "compile.sppcr"].decode())
    cases = {
        "basic": BASE,
        "continued": BASE.replace("100 102 104", "100\n102 104").replace(".5 1", ".5\n1"),
        "numeric_retry": "0\n" + BASE,
        "extra_values": BASE.replace("20 30", "20 30 99"),
        "drop_unseen": BASE.replace("4 8 2", "4 8 0").replace("10 15 5", "10 15 0"),
    }
    records = {}
    with tempfile.TemporaryDirectory(prefix="sppcr-interactive-") as tmp:
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
        scope=("Original interactive reader and conversion, with unused input arrays zeroed"),
        cases=records,
    )
    Path("tests/fixtures/sppcr_interactive.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
