"""Whether the resources a claim needs are present (for skipping, or failing under --strict)."""
from __future__ import annotations

import importlib.util
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
    if need == "cctbx":
        return _module("cctbx")
    if need == "sympy":
        return _module("sympy")
    if need == "lean":
        return shutil.which("lake") is not None and (paths.FORMAL / "lakefile.toml").is_file()
    if need == "xfel":
        return paths.XFEL_STREAM.is_file()
    if need == "sella":
        return shutil.which("CmdSella") is not None or shutil.which("cmdsella") is not None
    if need == "ncdist":
        return _module("ncdist") or _module("pycs6dist")
    if need == "lysozyme":
        return any(paths.DATA.glob("*lyso*"))
    raise KeyError(need)
