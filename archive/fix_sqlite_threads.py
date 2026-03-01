#!/usr/bin/env python3
import os
import re

def fix_sqlite_connections():
    """Patch SQLite connections to allow cross-thread usage"""
    patched = 0
    
    for root, dirs, files in os.walk('.'):
        if 'venv' in root or '__pycache__' in root or '.git' in root:
            continue
            
        for file in files:
            if not file.endswith('.py'):
                continue
                
            filepath = os.path.join(root, file)
            
            try:
                with open(filepath, 'r') as f:
                    content = f.read()
                
                original = content
                changes = []
                
                # Pattern 1: sqlite3.connect() without check_same_thread
                if 'sqlite3.connect(' in content and 'check_same_thread' not in content:
                    content = re.sub(
                        r'sqlite3\.connect\(([^)]+)\)',
                        r'sqlite3.connect(\1, check_same_thread=False)',
                        content
                    )
                    if content != original:
                        changes.append("Added check_same_thread=False")
                
                # Pattern 2: aiosqlite or async connections
                if 'sqlite3' in content and 'Thread' in content:
                    # Add connection recreation logic for threads
                    content = re.sub(
                        r'(def.*sync.*file.*\([^)]*\):)',
                        r'\1\n        import threading\n        if hasattr(self, "_conn") and threading.current_thread().ident != getattr(self, "_thread_id", None):\n            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)\n            self._thread_id = threading.current_thread().ident',
                        content,
                        flags=re.DOTALL
                    )
                
                if content != original:
                    with open(filepath, 'w') as f:
                        f.write(content)
                    print(f"✓ Patched: {filepath} ({', '.join(changes) if changes else 'sqlite3 fixes'})")
                    patched += 1
                    
            except Exception as e:
                continue
    
    return patched

def patch_local_cache_adapter():
    """Specifically target the LocalCacheAdapter mentioned in logs"""
    paths = [
        'memory/local_cache_adapter.py',
        'adapters/local_cache_adapter.py',
        'core/local_cache.py',
        'utils/local_cache.py'
    ]
    
    for path in paths:
        if os.path.exists(path):
            with open(path, 'r') as f:
                content = f.read()
            
            # Force check_same_thread=False on all connections
            if 'sqlite3.connect' in content:
                content = re.sub(
                    r'sqlite3\.connect\(["\']([^"\']+)["\'][^)]*\)',
                    r'sqlite3.connect("\1", check_same_thread=False)',
                    content
                )
                with open(path, 'w') as f:
                    f.write(content)
                print(f"✓ Fixed LocalCacheAdapter: {path}")
                return True
    return False

def patch_vector_store():
    """Fix vector store threading issues"""
    paths = [
        'vector_store/sqlite_local_v2.py',
        'brain/vector_store.py',
        'core/vector_store.py'
    ]
    
    for path in paths:
        if os.path.exists(path):
        import threading
        if hasattr(self, "_conn") and threading.current_thread().ident != getattr(self, "_thread_id", None):
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._thread_id = threading.current_thread().ident
            with open(path, 'r') as f:
                content = f.read()
            
            # Add thread safety
            if 'sqlite3' in content:
                content = re.sub(
                    r'(sqlite3\.connect\([^)]+)\)',
                    r'\1, check_same_thread=False)',
                    content
                )
                with open(path, 'w') as f:
                    f.write(content)
                print(f"✓ Fixed VectorStore: {path}")
                return True
    return False

if __name__ == "__main__":
    print("🔧 Patching SQLite threading issues...")
    patched = fix_sqlite_connections()
    patch_local_cache_adapter()
    patch_vector_store()
    print(f"\n✅ Done! Patched {patched} files.")
    print("\nRestart with: python main.py")
