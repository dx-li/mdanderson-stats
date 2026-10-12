"""Execute the original managed serializer on synthetic valid binary/TTE studies.

Requires Mono runtime/compiler supplied by the researcher, not a production dependency.
The environment must resolve the three original assemblies and Mono framework.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mono", required=True)
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--library-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    libraries = [
        args.library_dir / n for n in ("ARand.Design.dll", "ARandSimulation.dll", "ARand.Util.dll")
    ]
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in libraries}
    expected = {
        "ARand.Design.dll": "297cd5439341950cae09ed94e0e9f19bde0c1b9767cde3b5cd82b4fa3c76675d",
        "ARandSimulation.dll": "b42ec25836c7a0543c224b1778b73d90ca91163bd45c5a08c69fffc750a994a7",
        "ARand.Util.dll": "83c502f62d63594baa776c76492e97782edc6d34f48f57165dd0bb09b92af632",
    }
    if hashes != expected:
        raise ValueError("original managed library hashes do not match the reference archive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as work:
        executable = Path(work) / "probe.exe"
        subprocess.run(
            [
                args.mono,
                args.compiler,
                str(Path(__file__).with_name("ManagedStudyProbe.cs")),
                "-out:" + str(executable),
                *["-r:" + str(p.resolve()) for p in libraries],
            ],
            check=True,
        )
        result = subprocess.run(
            [args.mono, str(executable), str(args.output_dir.resolve())],
            check=True,
            text=True,
            capture_output=True,
        )
    assert result.stdout.splitlines() == ["binary valid=True ", "tte valid=True "], result.stdout
    outputs = {}
    for p in sorted(args.output_dir.glob("*.xml")):
        root = ET.fromstring(p.read_text())
        # Native getter uses the wall clock even after CurrentTime is explicitly assigned.
        root.find("./trial_details/current_time").text = "0"
        p.write_text(ET.tostring(root, encoding="unicode") + "\n")
        outputs[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    for family in ("binary", "tte"):
        assert outputs[family + ".xml"] == outputs[family + "-roundtrip.xml"]
    manifest = dict(
        profile="original-managed-serializer",
        libraries=hashes,
        outputs=outputs,
        validation=result.stdout.splitlines(),
        normalization=["trial_details/current_time replaced with numeric zero only"],
        limitations=[
            "Does not execute mixed-mode native trial engine.",
            "Synthetic history fields exhibit original managed setter/validator behavior.",
        ],
    )
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
