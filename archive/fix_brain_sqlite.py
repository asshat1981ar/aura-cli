#!/usr/bin/env python3
import re
import os

def fix_brain_py():
    """Fix the specific syntax error in brain.py"""
    filepath = 'memory/brain.py'
    
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return False
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    original = content
    
    # Fix the specific pattern: str(db_file_path, check_same_thread=False)
    # Should be: sqlite3.connect(str(db_file_path), check_same_thread=False)
    
    content = re.sub(
        r'str\(([^)]+),\s*check_same_thread=False\)',
        r'str(\1)',
        content
    )
    
    # Now add check_same_thread=False to sqlite3.connect if missing
    content = re.sub(
        r'sqlite3\.connect\((str\([^)]+)\)(?!,)',
        r'sqlite3.connect(\1, check_same_thread=False)',
        content
    )
    
    # Alternative pattern: sqlite3.connect(str(db_file_path)) 
    content = re.sub(
        r'sqlite3\.connect\((str\([^)]+)\)\)',
        r'sqlite3.connect(\1), check_same_thread=False)',
        content
    )
    
    if content != original:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"✓ Fixed: {filepath}")
        return True
    else:
        print("No changes made to brain.py")
        return False

def fix_momento_brain():
    """Fix momento_brain.py if it has similar issues"""
    filepath = 'memory/momento_brain.py'
    
    if not os.path.exists(filepath):
        return False
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Check if it calls super().__init__() which triggers brain.py
    # No direct fix needed here, the error is in brain.py
    
    return False

if __name__ == "__main__":
    fix_brain_py()
    print("\nNow run: python main.py")
