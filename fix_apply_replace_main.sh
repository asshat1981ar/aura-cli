#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

cd "$(dirname "$0")" 2>/dev/null || true

if [ ! -f "main.py" ]; then
  echo "ERROR: main.py not found. Run this from the repo root (the folder containing main.py)."
  exit 1
fi

TS="$(date +%Y%m%d_%H%M%S)"
cp -f main.py "main.py.bak_$TS"
echo "[+] Backup: main.py.bak_$TS"

python - <<'PY'
from pathlib import Path
import re

p = Path("main.py")
s = p.read_text(encoding="utf-8", errors="replace")

# Ensure imports needed by helper
need_imports = []
if not re.search(r"(?m)^from pathlib import Path\s*$", s):
    need_imports.append("from pathlib import Path")
if not re.search(r"(?m)^import os\s*$", s):
    need_imports.append("import os")

if need_imports:
    lines = s.splitlines(True)
    last_imp = 0
    for i, line in enumerate(lines[:250]):
        if line.startswith("import ") or line.startswith("from "):
            last_imp = i + 1
    for imp in need_imports:
        lines.insert(last_imp, imp + "\n")
        last_imp += 1
    s = "".join(lines)

# Inject helper function once
if "_ensure_file_and_maybe_overwrite" not in s:
    helper = r'''
def _ensure_file_and_maybe_overwrite(repo_root: str, file_path: str, old_code: str, new_code: str) -> bool:
    """
    Returns True if it handled the write (created/overwrote) and the caller should skip normal replace().
    Returns False if caller should attempt normal replace() semantics.
    """
    # Normalize path relative to repo root (your logs show repo-root based file resolution)
    abs_path = Path(repo_root) / file_path
    abs_path.parent.mkdir(parents=True, exist_ok=True)

    # If file doesn't exist -> create with new_code
    if not abs_path.exists():
        abs_path.write_text(new_code or "", encoding="utf-8")
        return True

    current = abs_path.read_text(encoding="utf-8", errors="ignore")

    # If file is empty-ish and old_code is empty/placeholder -> overwrite with new_code
    old = (old_code or "").strip()
    if not current.strip():
        placeholder = (not old) or ("existing" in old.lower()) or ("placeholder" in old.lower())
        if placeholder:
            abs_path.write_text(new_code or "", encoding="utf-8")
            return True

    return False
'''
    # Insert helper after imports (or near top)
    lines = s.splitlines(True)
    insert_at = 0
    for i, line in enumerate(lines[:300]):
        if line.startswith("import ") or line.startswith("from "):
            insert_at = i + 1
    lines.insert(insert_at, helper + "\n")
    s = "".join(lines)

# Now patch the apply/replace call-site.
#
# We search for the most common pattern in your project:
# something like: file_tools.replace(file_path, old_code, new_code, ...)
# and inject the guard right before it.
#
# This is intentionally conservative: patch first occurrence only.
pat = re.compile(r"(?m)^(?P<ind>\s*)(?P<call>\w+\.\w*replace\()\s*(?P<args>.*)$")

lines = s.splitlines(True)

patched = False
for i, line in enumerate(lines):
    # Find a line that looks like a replace call AND references old_code/new_code
    if "replace(" in line and ("old_code" in line or "old_code_str" in line) and ("new_code" in line or "new_code_str" in line):
        ind = re.match(r"^(\s*)", line).group(1)

        # Try to infer repo root var name. Common ones: repo_root, ROOT, here, etc.
        # Your logs show absolute paths under /data/data/.../aura_cli/aura-cli so main.py likely has a root var.
        # We'll default to "." if unknown.
        repo_root_expr = "repo_root" if "repo_root" in s else "os.getcwd()"

        guard = (
            f"{ind}# AURA patch: create missing files and overwrite empty files when old_code is placeholder\n"
            f"{ind}handled = _ensure_file_and_maybe_overwrite({repo_root_expr}, file_path, old_code, new_code)\n"
            f"{ind}if handled:\n"
            f"{ind}    log_json('INFO', 'file_created_or_overwritten', goal=current_goal, details={{'file': file_path, 'change_idx': change_idx}})\n"
            f"{ind}    continue\n"
        )

        lines.insert(i, guard)
        patched = True
        break

# If we didn't find a call-site, don't silently “succeed”
if not patched:
    raise SystemExit(
        "Could not find the replace(...) call-site referencing old_code/new_code in main.py. "
        "Search for the function where code changes are applied and patch manually."
    )

p.write_text("".join(lines), encoding="utf-8")
print("✅ Patched main.py apply/replace flow (create/overwrite support).")
PY

echo "[*] Compile check"
python -m py_compile main.py

echo
echo "[+] Done. Run:"
echo "    python main.py"
