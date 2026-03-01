#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

[ -f main.py ] || { echo "ERROR: run from repo root (main.py missing)"; exit 1; }

TS="$(date +%Y%m%d_%H%M%S)"
cp -f main.py "main.py.bak_$TS"
echo "[+] Backup: main.py.bak_$TS"

python - <<'PY'
from pathlib import Path
import re

p = Path("main.py")
s = p.read_text(encoding="utf-8", errors="replace")

def ensure_import(code: str, imp: str) -> str:
    if re.search(rf"(?m)^\s*{re.escape(imp)}\s*$", code):
        return code
    lines = code.splitlines(True)
    last = 0
    for i, line in enumerate(lines[:250]):
        if line.startswith("import ") or line.startswith("from "):
            last = i + 1
    lines.insert(last, imp + "\n")
    return "".join(lines)

s = ensure_import(s, "from pathlib import Path")

if "_aura_apply_change" not in s:
    helper = """
def _aura_apply_change(project_root, file_path, old_code, new_code, overwrite_file=False):
    \"""
    Resilient code application:
    - create file if missing
    - overwrite if empty
    - try replace_code
    - if old_code not found: append new_code once (unless overwrite_file=True)
    \"""
    full_path = Path(project_root) / file_path
    full_path.parent.mkdir(parents=True, exist_ok=True)

    if not full_path.exists():
        full_path.write_text(new_code or "", encoding="utf-8")
        return "created"

    current = full_path.read_text(encoding="utf-8", errors="ignore")

    if not current.strip():
        full_path.write_text(new_code or "", encoding="utf-8")
        return "overwrote_empty"

    try:
        replace_code(str(full_path), old_code, new_code, overwrite_file=overwrite_file)
        return "replaced"
    except OldCodeNotFoundError:
        if overwrite_file:
            full_path.write_text(new_code or "", encoding="utf-8")
            return "overwrote_forced"

        # append only if not already present (prevents infinite duplication)
        if (new_code or "").strip() and (new_code not in current):
            with full_path.open("a", encoding="utf-8") as f:
                f.write("\\n\\n# --- AURA APPEND (old_code not found) ---\\n")
                f.write(new_code)
                f.write("\\n# --- END AURA APPEND ---\\n")
            return "appended"

        raise
"""
    # insert helper after imports
    lines = s.splitlines(True)
    insert_at = 0
    for i, line in enumerate(lines[:300]):
        if line.startswith("import ") or line.startswith("from "):
            insert_at = i + 1
    lines.insert(insert_at, helper + "\n")
    s = "".join(lines)

# Regex-patch the first replace_code(...) call that targets project_root/file_path and passes overwrite_file
pattern = re.compile(
    r"replace_code\(\s*str\(\s*project_root\s*/\s*file_path\s*\)\s*,\s*old_code\s*,\s*new_code\s*,\s*overwrite_file\s*=\s*overwrite_file\s*\)",
    re.MULTILINE
)

replacement = (
    'outcome = _aura_apply_change(project_root, file_path, old_code, new_code, overwrite_file=overwrite_file)\n'
    '                                        log_json("INFO", "apply_change_outcome", goal=current_goal, details={"file": file_path, "change_idx": change_idx, "overwrite": overwrite_file, "outcome": outcome})'
)

s2, n = pattern.subn(replacement, s, count=1)
if n == 0:
    raise SystemExit("Could not regex-match replace_code(...) call. Paste the apply block around it and we’ll patch precisely.")
s = s2

p.write_text(s, encoding="utf-8")
print("✅ Patched main.py (regex): replace_code -> _aura_apply_change + outcome log.")
PY

python -m py_compile main.py
echo "[+] main.py compiles"
echo "[+] Run: python main.py"
