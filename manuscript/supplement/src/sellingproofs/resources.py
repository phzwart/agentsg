"""Whether the resources a claim needs are present (for skipping, or failing under --strict)."""
from __future__ import annotations

import importlib.util
import os
import shutil
from functools import lru_cache

from . import paths


def _module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


@lru_cache(maxsize=None)
def available(need: str) -> bool:
    if need == "slow":
        return True
    if need == "pdb":
        return paths.PDB_DB.is_file()
    if need == "cctbx":                                # importable, not merely present: cctbx
        if not _module("cctbx"):                       # refuses to load from some virtual
            return False                               # environments, as do the scripts run
        import subprocess, sys                         # in a subprocess by the reproduce claims
        r = subprocess.run([sys.executable, "-c", "import cctbx.uctbx"], capture_output=True)
        return r.returncode == 0
    if need == "sympy":
        return _module("sympy")
    if need == "lean":
        return shutil.which("lake") is not None and (paths.FORMAL / "lakefile.toml").is_file()
    if need == "xfel":
        return paths.XFEL_STREAM.is_file()
    if need == "sella":                                # compare_sella.py reads SELLA_BIN
        b = os.environ.get("SELLA_BIN")
        return bool(b and os.path.isfile(b)) or shutil.which("CmdSella") is not None
    if need == "ncdist":                               # a checkout of yayahjb/ncdist
        d = os.environ.get("NCDIST_DIR")
        return bool(d and os.path.isdir(d))
    if need == "lysozyme":
        return any(paths.DATA.glob("*lyso*"))
    raise KeyError(need)
