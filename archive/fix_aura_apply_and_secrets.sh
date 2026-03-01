#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

# Run from repo root (folder containing main.py)
if [ ! -f "main.py" ]; then
  echo "ERROR: main.py not found. cd into the repo root first."
  exit 1
fi

TS="$(date +%Y%m%d_%H%M%S)"
cp -f main.py "main.py.bak_$TS"
echo "[+] Backed up main.py -> main.py.bak_$TS"

# 1) Patch main.py: add safe fallback for file create + old_code_not_found
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

if "_aura_safe_apply_change" not in s:
    helper = """
def _aura_safe_apply_change(full_path, old_code, new_code, overwrite_file=False):
    \"""
    Safe wrapper:
    - create missing file + dirs
    - overwrite empty files
    - attempt replace_code
    - if old_code not found: append new_code (or overwrite if overwrite_file=True)
    \"""
    path_obj = Path(full_path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    if not path_obj.exists():
        path_obj.write_text(new_code or "", encoding="utf-8")
        return "created"

    current = path_obj.read_text(encoding="utf-8", errors="ignore")
    if not current.strip():
        path_obj.write_text(new_code or "", encoding="utf-8")
        return "overwrote_empty"

    try:
        replace_code(str(path_obj), old_code, new_code, overwrite_file=overwrite_file)
        return "replaced"
    except OldCodeNotFoundError:
        # If overwrite explicitly allowed, do it.
        if overwrite_file:
            path_obj.write_text(new_code or "", encoding="utf-8")
            return "overwrote_forced"

        # Otherwise: append if not already present (avoid endless duplication)
        if (new_code or "").strip() and (new_code not in current):
            with path_obj.open("a", encoding="utf-8") as f:
                f.write("\\n\\n# --- AURA APPEND (old_code not found) ---\\n")
                f.write(new_code)
                f.write("\\n# --- END AURA APPEND ---\\n")
            return "appended"

        # Nothing safe to do
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

# Replace the direct replace_code call in your apply loop with the safe wrapper + log outcome
target = "replace_code(str(project_root / file_path), old_code, new_code, overwrite_file=overwrite_file)"
if target not in s:
    raise SystemExit("Could not find the expected replace_code(...) call in main.py")

replacement = (
    "outcome = _aura_safe_apply_change(project_root / file_path, old_code, new_code, overwrite_file=overwrite_file)\n"
    "                                        log_json(\"INFO\", \"apply_change_outcome\", goal=current_goal, details={\"file\": file_path, \"change_idx\": change_idx, \"outcome\": outcome, \"overwrite\": overwrite_file})"
)

s = s.replace(target, replacement)

p.write_text(s, encoding="utf-8")
print("✅ Patched main.py: resilient apply (create/overwrite/append fallback).")
PY

python -m py_compile main.py
echo "[+] main.py compiles"

# 2) Remove/neutralize any prints that leak OPENAI_API_KEY in the repo (best-effort)
#    This comments out lines that contain both OPENAI_API_KEY and print(
echo "[*] Redacting obvious OPENAI_API_KEY prints (best-effort)…"
python - <<'PY'
from pathlib import Path
import re

changed = 0
for p in Path(".").rglob("*.py"):
    try:
        s = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        continue

    s2 = s

    # Comment out lines that print the key directly
    s2 = re.sub(r'(?m)^\s*print\((.*OPENAI_API_KEY.*)\)\s*$', r'# [REDACTED] print(\1)', s2)

# [REDACTED]     # Comment out lines that log "OPENAI_API_KEY value" or similar
# [REDACTED]     s2 = re.sub(r'(?m)^\s*.*OPENAI_API_KEY value.*$', r'# [REDACTED] \g<0>', s2)

    if s2 != s:
        p.write_text(s2, encoding="utf-8")
        changed += 1

print(f"✅ Redacted in {changed} file(s).")
PY

# 3) Safety scan: warn if key-looking strings remain
echo "[*] Scanning for sk-proj- strings (first 10 hits)…"
grep -R "sk-proj-" -n . --exclude-dir=__pycache__ | head -n 10 || true

echo
echo "[+] Done."
echo "Run: python main.py"
