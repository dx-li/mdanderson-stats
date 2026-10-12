"""Run original DSF managed DTOs/serializer with native CRT initialization disabled.

Requires Mono/mcs and pefile. Only a temporary kernel copy is patched; original
assets stay unchanged and are not distributed. This checks managed saved-file
contracts, not native construction or a complete Windows application session.
"""

import argparse
import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path

import pefile

HASHES = {
    "DSF_KernelDLL.dll": "f203c2f774ecabb390e58213076fabe02df15aa74c62df9861a7e2e0ac675a1e",
    "Utilities.dll": "77e3997f2375ba3d54e3185d7659dd5ef78a378f8c2f04ff9d506a0e7cfc3b64",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets", type=Path)
    parser.add_argument("--mono", default="mono")
    parser.add_argument("--compiler", default="mcs")
    parser.add_argument("--compiler-via-mono", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("tests/fixtures/dose-schedule-desktop-study.xml")
    )
    args = parser.parse_args()
    for name, digest in HASHES.items():
        if hashlib.sha256((args.assets / name).read_bytes()).hexdigest() != digest:
            raise ValueError("original asset hash mismatch: " + name)
    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)
        for name in HASHES:
            shutil.copyfile(args.assets / name, folder / name)
        kernel = folder / "DSF_KernelDLL.dll"
        pe = pefile.PE(str(kernel))
        raw = bytearray(kernel.read_bytes())
        offset = pe.get_offset_from_rva(0x1F990)
        raw[offset : offset + 2] = b"\x06\x2a"  # tiny managed method body: ret
        kernel.write_bytes(raw)
        driver = Path(__file__).parent / "dose_schedule_reference/study.cs"
        compiler = [args.mono, args.compiler] if args.compiler_via_mono else [args.compiler]
        subprocess.run(
            compiler
            + [
                f"-r:{kernel}",
                f"-r:{folder / 'Utilities.dll'}",
                f"-out:{folder / 'study.exe'}",
                str(driver.resolve()),
            ],
            check=True,
            capture_output=True,
        )
        result = subprocess.run(
            [args.mono, str(folder / "study.exe")], check=True, capture_output=True, text=True
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result.stdout)


if __name__ == "__main__":
    main()
