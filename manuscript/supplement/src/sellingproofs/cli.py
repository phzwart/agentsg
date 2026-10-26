"""python -m sellingproofs {ledger,run,pin}

    ledger [--format md|tex|json] [--out FILE]   every claim: z-number, place, kind, statement
    run [z52 z60 ...] [--fast] [--strict-needs]   run the claims (all, or those z-numbers)
    pin                                           library version, data hashes, git commit
"""
from __future__ import annotations

import argparse
import json
import re
import sys

from . import pin as _pin
from .registry import KINDS, NEEDS, load_all


def _tex(s: str) -> str:
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"),
                 ("#", r"\#"), ("{", r"\{"), ("}", r"\}"), ("~", r"\~{}"), ("^", r"\^{}")):
        s = s.replace(a, b)
    return s


def ledger(fmt: str = "md") -> str:
    claims = load_all()
    if fmt == "json":
        return json.dumps([c.__dict__ for c in claims], indent=1, ensure_ascii=False)
    if fmt == "tex":
        rows = []
        for z in sorted({c.z for c in claims}):
            cs = [c for c in claims if c.z == z]
            where = "; ".join(sorted({w for c in cs for w in c.where}))
            kinds = ", ".join(sorted({c.kind for c in cs}))
            needs = ", ".join(sorted({n for c in cs for n in c.needs}))
            rows.append(f"{z} & {_tex(where)} & {kinds}{' (' + needs + ')' if needs else ''} & {len(cs)} \\\\")
        return "\n".join(rows)
    out = ["# Ledger of claims", "",
           f"{len(claims)} checks of {len({c.z for c in claims})} numbered statements.", "",
           "| z | where | kind | needs | statement | check |", "|---|---|---|---|---|---|"]
    for c in claims:
        out.append(f"| {c.z} | {'; '.join(c.where)} | {c.kind} | {', '.join(c.needs)} | "
                   f"{c.statement.replace('|', '/')} | `{c.function.split('.')[-1]}` |")
    out += ["", "Kinds: " + "; ".join(f"**{k}** {v}" for k, v in KINDS.items()) + ".",
            "", "Needs: " + "; ".join(f"**{k}** {v}" for k, v in NEEDS.items()) + "."]
    return "\n".join(out)


def run(zs, fast=False, strict=False, extra=()) -> int:
    import pytest
    from . import paths
    ini = paths.SUPPLEMENT / "pytest.ini"
    args = [str(paths.PACKAGE / "claims"), "--rootdir", str(paths.SUPPLEMENT)]
    args += ["-c", str(ini)] if ini.is_file() else ["-p", "no:cacheprovider", "-q"]
    if zs:
        args += ["-m", " or ".join(zs)]
    if fast:
        args.append("--fast")
    if strict:
        args.append("--strict-needs")
    return pytest.main(args + list(extra))


def show_pin() -> int:
    head, clean = _pin.git_state()
    ok = True
    print(f"repository  {_pin.REPOSITORY}")
    print(f"commit      {head}{'' if clean else ' (uncommitted changes)'}")
    v = _pin.agentsg_version()
    print(f"agentsg     {v} (pinned {_pin.AGENTSG_VERSION})")
    ok &= v == _pin.AGENTSG_VERSION
    for rel, want in _pin.SHA256.items():
        got = _pin.sha256(rel)
        print(f"{'ok' if got == want else 'MISMATCH':8s}    {rel}")
        ok &= got == want
    return 0 if ok else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m sellingproofs", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    l = sub.add_parser("ledger"); l.add_argument("--format", default="md", choices=("md", "tex", "json"))
    l.add_argument("--out")
    sub.add_parser("run", help="run [z52 ...] [--fast] [--strict-needs] [pytest options]")
    sub.add_parser("pin")
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["run"]:
        rest = argv[1:]
        zs = [x for x in rest if re.fullmatch(r"z\d\d", x)]
        opts = [x for x in rest if x not in zs and x not in ("--fast", "--strict-needs")]
        return run(zs, "--fast" in rest, "--strict-needs" in rest, opts)
    a = p.parse_args(argv)
    if a.cmd == "ledger":
        txt = ledger(a.format)
        if a.out:
            open(a.out, "w").write(txt + "\n")
        else:
            print(txt)
        return 0
    return show_pin()


if __name__ == "__main__":
    sys.exit(main())
