#!/data/data/com.termux/files/usr/bin/env python3
"""
AURA CLI JSON Parser Fix for Termux
Patches the model response parsing to handle markdown-wrapped JSON
"""

import os
import sys
import re
import shutil
from core.file_tools import _aura_safe_loads # Import the robust JSON loader

def clean_and_patch_file(filepath):
    print(f"[INFO] Checking: {filepath}")
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Skip if already patched (check for the _aura_safe_loads import line)
    if 'from core.file_tools import _aura_safe_loads' in content:
        print(f"[SKIP] Already patched: {filepath}")
        return False
    
    # Look for json.loads usage
    if 'json.loads' not in content:
        return False
    
    # Create backup
    backup = filepath + '.backup'
    shutil.copy2(filepath, backup)
    print(f"[BACKUP] Created: {backup}")
    
    # Add the _aura_safe_loads import at the appropriate place
    lines = content.split('\n')
    insert_at = 0
    for i, line in enumerate(lines):
        if line.strip().startswith(('import ', 'from ')):
            insert_at = i + 1
    
    lines.insert(insert_at, 'from core.file_tools import _aura_safe_loads # Injected by fix_aura_json.py')
    
    # Replace problematic json.loads calls
    new_lines = []
    for line in lines:
        # Replace json.loads calls that handle model responses
        if 'json.loads' in line and any(x in line.lower() for x in ['response', 'raw', 'text', 'content']):
            # Simple replacement: json.loads(var) -> _aura_safe_loads(var, "model")
            line = re.sub(
                r'json\.loads\(([^)]+)\)',
                r'_aura_safe_loads(\1, "model_response")',
                line
            )
        new_lines.append(line)
    
    # Write back
    with open(filepath, 'w') as f:
        f.write('\n'.join(new_lines))
    
    print(f"[PATCHED] {filepath}")
    return True

def main():
    print("=" * 50)
    print("AURA CLI JSON Parser Fix")
    print("=" * 50)
    
    # Find Python files in current directory
    patched = 0
    for root, dirs, files in os.walk('.'):
        # Skip hidden and cache dirs
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'venv', 'env']]
        
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                try:
                    if clean_and_patch_file(filepath):
                        patched += 1
                except Exception as e:
                    print(f"[ERROR] {filepath}: {e}")
    
    print(f"\n[Patched {patched} files]")
    print("Run: python main.py")

if __name__ == "__main__":
    main()
