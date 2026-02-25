#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

if [ ! -f "main.py" ]; then
  echo "Run this from the repo root (where main.py lives)"
  exit 1
fi

TS=$(date +%Y%m%d_%H%M%S)
cp main.py main.py.bak_$TS
echo "[+] Backup created: main.py.bak_$TS"

python - <<'PY'
from pathlib import Path
import re

p = Path("main.py")
s = p.read_text(encoding="utf-8")

if "_safe_apply_change" in s:
    print("Already patched.")
    raise SystemExit

helper = '''
def _safe_apply_change(full_path, old_code, new_code, overwrite_file=False):
    """
    Safe wrapper around replace_code:
    - Creates file if missing
    - Overwrites empty files
    - Falls back to replace_code otherwise
    """
    path_obj = Path(full_path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    if not path_obj.exists():
        path_obj.write_text(new_code or "", encoding="utf-8")
        return

    current = path_obj.read_text(encoding="utf-8", errors="ignore")

    if not current.strip():
        path_obj.write_text(new_code or "", encoding="utf-8")
        return

    replace_code(str(path_obj), old_code, new_code, overwrite_file=overwrite_file)
'''

# Insert helper after imports
lines = s.splitlines(True)
insert_at = 0
for i, line in enumerate(lines):
    if line.startswith("import ") or line.startswith("from "):
        insert_at = i + 1

lines.insert(insert_at, helper + "\n")
s = "".join(lines)

# Replace the direct replace_code call
s = s.replace(
    'replace_code(str(project_root / file_path), old_code, new_code, overwrite_file=overwrite_file)',
    '_safe_apply_change(project_root / file_path, old_code, new_code, overwrite_file)'
)

p.write_text(s, encoding="utf-8")
print("Patched main.py safely.")
PY

python -m py_compile main.py
echo "[+] Compile OK"
echo "Run: python main.py"
