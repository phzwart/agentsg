"""Every test in this package is a registered claim; a claim whose resources are absent is
skipped, or fails under ``--strict-needs``."""
from __future__ import annotations

import pytest

from sellingproofs.registry import KINDS, NEEDS
from sellingproofs.resources import available


def pytest_addoption(parser):
    parser.addoption("--strict-needs", action="store_true",
                     help="fail, rather than skip, a claim whose data or programs are absent")
    parser.addoption("--fast", action="store_true", help="skip the slow claims")


def pytest_configure(config):
    config.addinivalue_line("markers", "claim: a registered claim of the paper")
    for z in range(100):
        config.addinivalue_line("markers", f"z{z:02d}: claim z{z:02d}")
    for k, v in KINDS.items():
        config.addinivalue_line("markers", f"{k}: {v}")
    for k, v in NEEDS.items():
        config.addinivalue_line("markers", f"{k}: needs {v}")


def pytest_collection_modifyitems(config, items):
    strict = config.getoption("--strict-needs")
    for item in items:
        c = getattr(getattr(item, "function", None), "__claim__", None)
        if c is None:
            raise pytest.UsageError(f"{item.nodeid} is not registered with @claim")
        if config.getoption("--fast") and "slow" in c.needs:
            item.add_marker(pytest.mark.skip(reason="slow (--fast)"))
            continue
        missing = [n for n in c.needs if not available(n)]
        if missing and not strict:
            item.add_marker(pytest.mark.skip(reason="needs " + ", ".join(NEEDS[n] for n in missing)))
