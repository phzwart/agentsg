"""Check that SellingClosure/Data.lean is exactly what gen_data.py produces from the
deposited table manuscript/supplement/benchmark/universal_operators_columns.json (by SHA-256 and by
regeneration).  Exit status 0 when they agree."""
import hashlib, pathlib, subprocess, sys, tempfile, shutil
here = pathlib.Path(__file__).resolve().parent
data = here / "SellingClosure" / "Data.lean"
src = here.parent / "benchmark" / "universal_operators_columns.json"
sha = hashlib.sha256(src.read_bytes()).hexdigest()
text = data.read_text()
ok_hash = f'def sourceSHA256 : String := "{sha}"' in text
backup = data.read_bytes()
subprocess.run([sys.executable, str(here / "gen_data.py")], check=True, capture_output=True)
same = data.read_bytes() == backup
data.write_bytes(backup)
print(f"source SHA-256 {sha}: {'recorded' if ok_hash else 'NOT recorded'}; regeneration {'identical' if same else 'DIFFERS'}")
sys.exit(0 if ok_hash and same else 1)
