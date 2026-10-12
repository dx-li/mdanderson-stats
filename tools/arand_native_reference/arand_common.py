"""Local original artifact validation; no download or executable launch."""

import hashlib
import os
from pathlib import Path

SHA256 = "a11d60157236bdf2e3395de39909dbb029b22bae52bcee5dd6d8ee4eebb3a6e3"


def library_path():
    path = Path(os.environ["ARAND_REFERENCE_LIBRARY"])
    expected = "a11d60157236bdf2e3395de39909dbb029b22bae52bcee5dd6d8ee4eebb3a6e3"
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"Original ARand.Calcs SHA256 mismatch: {actual}")
    return str(path)
