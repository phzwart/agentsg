"""Extract verbatim code evidence for each concept anchor (module, symbol)."""
from __future__ import annotations
import ast, hashlib, json, sys
from pathlib import Path

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("../agentsg/agentsg/src")
sys.path.insert(0, str(Path(__file__).parent))
from concepts import CONCEPTS  # noqa: E402
from quotes import first_sentence  # noqa: E402

REPO_URL = "https://github.com/phzwart/agentsg/blob/main/agentsg/src/"


def first_doc_line(node) -> str | None:
    d = ast.get_docstring(node, clean=False)
    if not d:
        return None
    sentence = first_sentence(d)
    return sentence or None


def find_symbol(tree, symbol):
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)) and n.name == symbol:
            return n
    return None


def main():
    files: dict[str, dict] = {}
    out = []
    problems = []
    ids = {c["id"] for c in CONCEPTS}
    for c in CONCEPTS:
        for rel, tgt in c["rels"]:
            if tgt not in ids:
                problems.append(f"{c['id']}: relation target {tgt} unknown")
        for module, symbol in c["anchors"]:
            if not module.endswith(".py"):
                problems.append(f"{c['id']}: anchor module {module} not a .py path")
                continue
            path = SRC / module
            if not path.exists():
                problems.append(f"{c['id']}: {module} missing")
                continue
            if module not in files:
                raw = path.read_bytes()
                files[module] = {"sha256": hashlib.sha256(raw).hexdigest(),
                                 "text": raw.decode("utf-8"),
                                 "tree": ast.parse(raw.decode("utf-8"))}
            f = files[module]
            node = f["tree"] if symbol == "" else find_symbol(f["tree"], symbol)
            if node is None:
                problems.append(f"{c['id']}: symbol {symbol} not in {module}")
                continue
            quote = first_doc_line(node)
            if quote is None:
                problems.append(f"{c['id']}: {module}:{symbol or '<module>'} has no docstring")
                continue
            doc = ast.get_docstring(node, clean=False) or ""
            collapsed = " ".join(doc.split())
            body = quote[:-3].rstrip() if quote.endswith("...") else quote
            if not collapsed.startswith(body):
                problems.append(f"{c['id']}: quote is not the docstring prefix in {module}")
                continue
            lineno = node.lineno
            if node.body and isinstance(node.body[0], ast.Expr):
                lineno = getattr(node.body[0], "lineno", lineno)
            out.append({"concept": c["id"], "module": module, "symbol": symbol,
                        "kind": "module" if symbol == "" else type(node).__name__,
                        "line": lineno, "exact": quote,
                        "sha256": f["sha256"], "uri": REPO_URL + module})
    json.dump({"anchors": out,
               "files": {m: f["sha256"] for m, f in files.items()}},
              open(Path(__file__).parent / "anchors.json", "w"), indent=1)
    print(len(CONCEPTS), "concepts;", len(out), "anchors;", len(files), "files;", len(problems), "problems")
    for p in problems:
        print("  -", p)


if __name__ == "__main__":
    main()
