"""What the supplement is pinned to.  The claims call the agentsg library of the same
repository; the paper's numbers hold for this library version and these data files, whose
SHA-256 hashes are checked by claim z00.  The deposited archive (Zenodo) is a tagged
commit of https://github.com/phzwart/agentsg; ``python -m sellingproofs pin`` prints the
commit of the checkout in use and whether it is clean."""
from __future__ import annotations

import hashlib
import subprocess

from . import paths

REPOSITORY = "https://github.com/phzwart/agentsg"
AGENTSG_VERSION = "0.5.0"
SHA256 = {
    # the 6096 operators of U as superbase rows (the library's copy)
    "agentsg/src/agentsg/cell/data/selling_operators.json":
        "2d5e46a42db4184b98e76254dbe7265fb892b88905fcc5920905ec1ca02b724f",
    # the same operators as column operators with determinants (the deposited table)
    "manuscript/supplement/benchmark/universal_operators_columns.json":
        "29f6f0269fe25aae123691c9baa16425e460616955704450fa8208e0bc38b024",
}


def sha256(rel: str) -> str:
    return hashlib.sha256((paths.REPO / rel).read_bytes()).hexdigest()


def git_state() -> tuple[str, bool]:
    """(commit, clean) of the checkout, or ("unknown", False) outside git."""
    try:
        head = subprocess.run(["git", "-C", str(paths.REPO), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(paths.REPO), "status", "--porcelain",
                                "--untracked-files=no"], capture_output=True, text=True,
                               check=True).stdout.strip()
        return head, not dirty
    except (OSError, subprocess.CalledProcessError):
        return "unknown", False


def agentsg_version() -> str:
    """the version of the agentsg library in use: its installed metadata, or the
    repository's pyproject.toml when it is used from the source tree."""
    import re
    try:
        import agentsg
        src = paths.REPO / "agentsg" / "src"
        if src in __import__("pathlib").Path(agentsg.__file__).resolve().parents:
            m = re.search(r'^version = "([^"]+)"', (paths.REPO / "pyproject.toml").read_text(), re.M)
            return m.group(1) if m else "unknown"
        from importlib.metadata import version
        return version("agentsg")
    except Exception:
        return "unknown"
