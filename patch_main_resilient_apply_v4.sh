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
src = p.read_text(encoding="utf-8", errors="replace")
lines = src.splitlines(True)

def ensure_import(imp: str):
    global lines
    for ln in lines[:250]:
        if ln.strip() == imp:
            return
    last_imp = 0
    for i, ln in enumerate(lines[:300]):
        if ln.startswith("import ") or ln.startswith("from "):
            last_imp = i + 1
    lines.insert(last_imp, imp + "\n")

ensure_import("from pathlib import Path")

# Insert helper once
if not any("def _aura_apply_change(" in ln for ln in lines):
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

        # Append only once to avoid infinite duplication
        if (new_code or "").strip() and (new_code not in current):
            with full_path.open("a", encoding="utf-8") as f:
                f.write("\\n\\n# --- AURA APPEND (old_code not found) ---\\n")
                f.write(new_code)
                f.write("\\n# --- END AURA APPEND ---\\n")
            return "appended"

        raise
"""
    # after imports
    insert_at = 0
    for i, ln in enumerate(lines[:300]):
        if ln.startswith("import ") or ln.startswith("from "):
            insert_at = i + 1
    lines.insert(insert_at, helper + "\n")

# Find applying_code_change log line
anchor = None
for i, ln in enumerate(lines):
    if 'event' in ln and 'applying_code_change' in ln:
        anchor = i
        break
    if 'log_json' in ln and 'applying_code_change' in ln:
        anchor = i
        break

if anchor is None:
    raise SystemExit("Could not find applying_code_change log in main.py")

# In next 20 lines, find first replace_code( and replace that *call* (multi-line safe)
start = anchor
end = min(len(lines), anchor + 40)

call_start = None
for i in range(start, end):
    if "replace_code" in lines[i] and "(" in lines[i]:
        call_start = i
        break

if call_start is None:
    raise SystemExit("Could not find replace_code(...) call near applying_code_change block")

# If call spans multiple lines, find its end by balancing parens
paren = 0
call_end = call_start
found_open = False
for j in range(call_start, end):
    for ch in lines[j]:
        if ch == "(":
            paren += 1
            found_open = True
        elif ch == ")":
            paren -= 1
    if found_open and paren <= 0:
        call_end = j
        break

indent = re.match(r"^(\s*)", lines[call_start]).group(1)

replacement = [
    f'{indent}outcome = _aura_apply_change(project_root, file_path, old_code, new_code, overwrite_file=overwrite_file)\n',
    f'{indent}log_json("INFO", "apply_change_outcome", goal=current_goal, details={{"file": file_path, "change_idx": change_idx, "overwrite": overwrite_file, "outcome": outcome}})\n'
]

# Replace the original replace_code(...) call block
lines[call_start:call_end+1] = replacement

p.write_text("".join(lines), encoding="utf-8")
print("✅ Patched main.py: replace_code(...) replaced with resilient _aura_apply_change(...).")
PY

python -m py_compile main.py
echo "[+] main.py compiles"
echo "[+] Run: python main.py"
