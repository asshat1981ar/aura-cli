#!/data/data/com.termux/files/usr/bin/bash
# fix_all_aura_issues.sh
# Usage:
#   cd ~/aura_cli/aura-cli   (or your repo root)
#   bash fix_all_aura_issues.sh
#
# What it does (with backups):
# [REDACTED] # - Stops printing OPENAI_API_KEY value (redacts / removes obvious debug prints)
# - Fixes HybridClosedLoop prompt formatting (kills KeyError '"DEFINE"')
# - Makes JSON parsing tolerate ```json fences
# - Ensures core/hybrid_loop.py imports json + GitToolsError
# - Adds GitTools.stash/stash_pop/stash_apply if missing + normalizes tabs->spaces
# - Fixes Brain DB path to ~/.aura/brain.sqlite (writable) + serializes dict/list to JSON
# - Patches the “replace/apply” module to create missing files/dirs and overwrite empty files
# - Adds .gitignore for pyc/bak/leaked artifacts
# - Scans for leaked OpenAI keys and warns (does NOT rotate keys; you must do that)

set -euo pipefail

ROOT="$(pwd)"
TS="$(date +%Y%m%d_%H%M%S)"

echo "[*] Repo root: $ROOT"
echo "[*] Timestamp: $TS"

backup() {
  local f="$1"
  [ -f "$f" ] || return 0
  cp -f "$f" "$f.bak_$TS"
}

py_patch() {
python - <<'PY'
import os, re, json
from pathlib import Path
from datetime import datetime

ROOT = Path(".").resolve()
TS = os.environ.get("TS") or datetime.now().strftime("%Y%m%d_%H%M%S")

def backup(p: Path):
    if p.exists():
        p.write_text(p.read_text(encoding="utf-8"), encoding="utf-8")  # ensure decodable
        bak = p.with_suffix(p.suffix + f".bak_{TS}")
        bak.write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
        return bak
    return None

def normalize_tabs(p: Path):
    if not p.exists(): return
    s = p.read_text(encoding="utf-8", errors="replace")
    s2 = s.replace("\t", "    ")
    if s2 != s:
        backup(p)
        p.write_text(s2, encoding="utf-8")

def ensure_import(s: str, imp_line: str) -> str:
    if re.search(rf"(?m)^\s*{re.escape(imp_line)}\s*$", s):
        return s
    lines = s.splitlines(True)
    last_imp = 0
    for i, line in enumerate(lines[:250]):
        if line.startswith("import ") or line.startswith("from "):
            last_imp = i + 1
    lines.insert(last_imp, imp_line + "\n")
    return "".join(lines)

def patch_hybrid_loop():
    p = ROOT / "core" / "hybrid_loop.py"
    if not p.exists():
        print("[!] core/hybrid_loop.py not found; skipping")
        return

    normalize_tabs(p)
    s = p.read_text(encoding="utf-8", errors="replace")
    bak = backup(p)

    # Ensure imports
    s = ensure_import(s, "import json")
    s = ensure_import(s, "import re")
    # Import GitToolsError if referenced or exception handler exists
    if ("GitToolsError" in s) and ("from core.git_tools import GitToolsError" not in s):
        s = ensure_import(s, "from core.git_tools import GitToolsError")

    # Ensure fence stripper exists
    if "def _strip_code_fences" not in s:
        helper = (
            "\n"
            "_FENCE_RE = re.compile(r\"^\\s*```(?:json)?\\s*|\\s*```\\s*$\", re.IGNORECASE)\n"
            "\n"
            "def _strip_code_fences(text: str) -> str:\n"
            "    if not isinstance(text, str):\n"
            "        return \"\"\n"
            "    return _FENCE_RE.sub(\"\", text).strip()\n"
            "\n"
        )
        # Insert helper after imports
        lines = s.splitlines(True)
        last_imp = 0
        for i, line in enumerate(lines[:250]):
            if line.startswith("import ") or line.startswith("from "):
                last_imp = i + 1
        lines.insert(last_imp, helper)
        s = "".join(lines)

    # Fix prompt building: .format(...) -> .replace(...)
    pat = re.compile(
        r'(?m)^(?P<ind>\s*)prompt\s*=\s*self\._bootstrap_prompt_template\.format\s*GOAL\s*=\s*goal\s*,\s*STATE\s*=\s*state\s*\s*$'
    )
    def repl(m):
        ind = m.group("ind")
        return (
            f"{ind}prompt = (\n"
            f"{ind}    self._bootstrap_prompt_template\n"
            f'{ind}    .replace("{{GOAL}}", str(goal))\n'
            f'{ind}    .replace("{{STATE}}", str(state))\n'
            f"{ind})"
        )
    s, _ = pat.subn(repl, s, count=1)

    # Fix JSON parsing: json.loads(raw_response) -> json.loads(_strip_code_fences(raw_response))
    s = re.sub(r'json\.loads\s*raw_response\s*', r'json.loads(_strip_code_fences(raw_response))', s)

    p.write_text(s, encoding="utf-8")
    print(f"[+] Patched core/hybrid_loop.py (backup: {bak.name if bak else 'none'})")

def patch_git_tools():
    p = ROOT / "core" / "git_tools.py"
    if not p.exists():
        print("[!] core/git_tools.py not found; skipping")
        return

    normalize_tabs(p)
    s = p.read_text(encoding="utf-8", errors="replace")
    bak = backup(p)

    # If stash already exists, don’t re-add
    if re.search(r"(?m)^\s*def\s+stash\s*", s):
        p.write_text(s, encoding="utf-8")
        print(f"[=] GitTools.stash already exists (backup: {bak.name if bak else 'none'})")
        return

    # Insert methods after 'class GitTools' line
    lines = s.splitlines(True)
    cls_idx = None
    for i, line in enumerate(lines):
        if re.match(r"^class\s+GitTools\b", line):
            cls_idx = i
            break
    if cls_idx is None:
        print("[!] class GitTools not found; skipping stash injection")
        return

    methods = """
    def stash(self, message: str = "AURA stash"):
        \"\"\"Create a stash (includes untracked).\"\"\"
        try:
            return self.repo.git.stash("push", "-u", "-m", message)
        except Exception as e:
            raise GitToolsError(f"git stash failed: {e}")

    def stash_pop(self):
        \"\"\"Pop latest stash.\"\"\"
        try:
            return self.repo.git.stash("pop")
        except Exception as e:
            raise GitToolsError(f"git stash pop failed: {e}")

    def stash_apply(self, ref: str = "stash@{0}"):
        \"\"\"Apply stash without dropping it.\"\"\"
        try:
            return self.repo.git.stash("apply", ref)
        except Exception as e:
            raise GitToolsError(f"git stash apply failed: {e}")
"""
    lines.insert(cls_idx + 1, methods)
    s2 = "".join(lines).replace("\t", "    ")
    p.write_text(s2, encoding="utf-8")
    print(f"[+] Patched core/git_tools.py (added stash methods) (backup: {bak.name if bak else 'none'})")

def patch_brain():
    p = ROOT / "memory" / "brain.py"
    if not p.exists():
        print("[!] memory/brain.py not found; skipping")
        return

    normalize_tabs(p)
    s = p.read_text(encoding="utf-8", errors="replace")
    bak = backup(p)

    # Ensure imports
    s = ensure_import(s, "import os")
    s = ensure_import(s, "import json")
    if "from pathlib import Path" not in s:
        s = ensure_import(s, "from pathlib import Path")

    # Ensure remember() serializes non-strings
    m = re.search(r"(?m)^(\s*)def\s+remember\s*\((.*?):\s*$", s)
    if m and "json.dumps" not in s:
        base_indent = m.group(1) + "    "
        inject = (
            f"{base_indent}# Coerce structured objects to JSON for SQLite storage\n"
            f"{base_indent}if not isinstance(text, str):\n"
            f"{base_indent}    text = json.dumps(text, ensure_ascii=False, default=str)\n\n"
        )
        lines = s.splitlines(True)
        for i, line in enumerate(lines):
            if re.match(r"^\s*def\s+remember\s*", line):
                lines.insert(i + 1, inject)
                s = "".join(lines)
                break

    # Force DB path to ~/.aura/brain.sqlite (writable) by injecting into __init__
    init = re.search(r"(?m)^(\s*)def\s+__init__\s*\(.*:\s*$", s)
    if init and "brain.sqlite" not in s:
        ind = init.group(1) + "    "
        block = (
            f"{ind}# Always store brain DB in a writable user directory\n"
            f"{ind}base_dir = Path(os.path.expanduser('~/.aura'))\n"
            f"{ind}base_dir.mkdir(parents=True, exist_ok=True)\n"
            f"{ind}db_path = base_dir / 'brain.sqlite'\n\n"
        )
        lines = s.splitlines(True)
        for i, line in enumerate(lines):
            if re.match(r"^\s*def\s+__init__\s*", line):
                lines.insert(i + 1, block)
                s = "".join(lines)
                break

    # Replace first sqlite3.connect(...) to connect(str(db_path)) if db_path exists in file
    if "db_path" in s:
        s = re.sub(r"sqlite3\.connect\((.*?)", "sqlite3.connect(str(db_path))", s, count=1)

    p.write_text(s, encoding="utf-8")
    print(f"[+] Patched memory/brain.py (backup: {bak.name if bak else 'none'})")

def patch_apply_replace_module():
    # Find likely file by searching for the logged event strings
    candidates = []
    for p in ROOT.rglob("*.py"):
        try:
            t = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if "old_code_not_found" in t or "file_not_found_for_replace" in t:
            candidates.append(p)

    if not candidates:
        print("[!] No apply/replace module found via log strings; skipping")
        return

    # Pick the smallest matching file (usually the handler)
    candidates.sort(key=lambda x: x.stat().st_size)
    p = candidates[0]
    normalize_tabs(p)
    s = p.read_text(encoding="utf-8", errors="replace")
    bak = backup(p)

    # Heuristic patch: ensure directories + create file if missing + overwrite empty files
    s = ensure_import(s, "from pathlib import Path")
    s = ensure_import(s, "import os")

    # Patch a function named replace(...) if present
    if re.search(r"(?m)^\s*def\s+replace\s*", s):
        # Insert "ensure file exists" near top of replace()
        def_pat = re.compile(r"(?m)^(\s*)def\s+replace\s*\((.*?):\s*$")
        m = def_pat.search(s)
        if m:
            ind = m.group(1) + "    "
            guard = (
                f"{ind}# AURA patch: create missing files/dirs and allow overwrite of empty files\n"
                f"{ind}path_obj = Path(file_path)\n"
                f"{ind}path_obj.parent.mkdir(parents=True, exist_ok=True)\n"
                f"{ind}if not path_obj.exists():\n"
                f"{ind}    path_obj.write_text(new_string, encoding='utf-8')\n"
                f"{ind}    return True\n"
                f"{ind}current = path_obj.read_text(encoding='utf-8', errors='ignore')\n"
                f"{ind}if (not current.strip()) and (old_string is None or str(old_string).strip()=='' or 'existing' in str(old_string).lower()):\n"
                f"{ind}    path_obj.write_text(new_string, encoding='utf-8')\n"
                f"{ind}    return True\n\n"
            )
            # Insert guard right after function line
            lines = s.splitlines(True)
            for i, line in enumerate(lines):
                if re.match(r"^\s*def\s+replace\s*", line):
                    lines.insert(i + 1, guard)
                    s = "".join(lines)
                    break
            p.write_text(s, encoding="utf-8")
            print(f"[+] Patched replace() in {p} (backup: {bak.name if bak else 'none'})")
            return

    # If no replace(), still add a note
    p.write_text(s, encoding="utf-8")
    print(f"[=] Found candidate {p} but no replace() function; manual patch may be needed (backup: {bak.name if bak else 'none'})")

def redact_api_key_prints():
    # Remove obvious debug prints that leak OPENAI_API_KEY
    patterns = [
        r'(?m)^\s*print\(.+OPENAI_API_KEY.+\s*$',
        r'(?m)^\s*console\.print.+OPENAI_API_KEY.+\s*$',
        r'(?m)^\s*logger\.\w+.+OPENAI_API_KEY.+\s*$',
    ]
    changed = 0
    for p in ROOT.rglob("*.py"):
        s = p.read_text(encoding="utf-8", errors="ignore")
        s2 = s
        for pat in patterns:
            s2 = re.sub(pat, "    # [REDACTED] removed secret logging", s2)
        if s2 != s:
            backup(p)
            p.write_text(s2, encoding="utf-8")
            changed += 1
    print(f"[+] Redacted OPENAI_API_KEY prints in {changed} file(s) (best-effort)")

def main():
    redact_api_key_prints()
    patch_git_tools()
    patch_hybrid_loop()
    patch_brain()
    patch_apply_replace_module()

    # normalize tabs across project
    for p in ROOT.rglob("*.py"):
        normalize_tabs(p)

main()
PY
}

export TS="$TS"
py_patch

echo "[*] Writing .gitignore (append-safe)"
if [ ! -f .gitignore ]; then
  touch .gitignore
fi
backup .gitignore
grep -q '^__pycache__/' .gitignore || cat >> .gitignore <<'EOF'

# Python bytecode / caches
__pycache__/
*.pyc

# Backups created during patching
*.bak
*.bak_*
EOF

echo "[*] Scanning for leaked OpenAI keys (sk-...) in repo (best-effort)"
if grep -R "sk-proj-" -n . >/dev/null 2>&1; then
  echo "[!] FOUND sk-proj- style key material in files:"
  grep -R "sk-proj-" -n . | head -n 20
  echo "[!] Rotate/revoke the key in your OpenAI dashboard NOW."
else
  echo "[+] No sk-proj- strings found (does not guarantee safety if logged elsewhere)."
fi

echo "[*] Compile gate: python -m py_compile on all .py"
python -m py_compile $(find . -name "*.py")

echo
echo "[+] Done."
echo "Next run:"
echo "  python aura_cli.py   (or python main.py depending on your entrypoint)"
echo
echo "[!] Reminder: if you ever printed your API key, assume it’s compromised and rotate it."
