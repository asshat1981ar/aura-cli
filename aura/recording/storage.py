"""Storage backend for recordings."""

from pathlib import Path
from typing import List, Optional

import yaml

from .models import Recording


class RecordingStorage:
    """Store and retrieve recordings."""
    
    DEFAULT_DIR = Path("~/.aura/recordings").expanduser()
    
    def __init__(self, directory: Optional[Path] = None):
        self.directory = directory or self.DEFAULT_DIR
        self.directory.mkdir(parents=True, exist_ok=True)
    
    def _get_file_path(self, name: str) -> Path:
        """Get file path for a recording."""
        # Sanitize name for filesystem
        safe_name = "".join(c for c in name if c.isalnum() or c in "-_ ").rstrip()
        safe_name = safe_name.replace(" ", "_")
        return self.directory / f"{safe_name}.yaml"
    
    async def save(self, recording: Recording) -> Path:
        """Save a recording to storage."""
        file_path = self._get_file_path(recording.name)
        
        data = recording.to_dict()
        
        with open(file_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        
        return file_path
    
    async def load(self, name: str) -> Optional[Recording]:
        """Load a recording by name."""
        file_path = self._get_file_path(name)
        
        if not file_path.exists():
            # Try with ID
            for f in self.directory.glob("*.yaml"):
                try:
                    with open(f) as fp:
                        data = yaml.safe_load(fp)
                    if data.get("id") == name or data.get("name") == name:
                        return Recording.from_dict(data)
                except Exception:
                    continue
            return None
        
        try:
            with open(file_path) as f:
                data = yaml.safe_load(f)
            return Recording.from_dict(data)
        except Exception:
            return None
    
    async def list(self) -> List[Recording]:
        """List all recordings."""
        recordings = []
        
        for file_path in self.directory.glob("*.yaml"):
            try:
                with open(file_path) as f:
                    data = yaml.safe_load(f)
                recordings.append(Recording.from_dict(data))
            except Exception:
                continue
        
        return recordings
    
    async def delete(self, name: str) -> bool:
        """Delete a recording."""
        file_path = self._get_file_path(name)
        
        if file_path.exists():
            file_path.unlink()
            return True
        
        # Try to find by ID
        for f in self.directory.glob("*.yaml"):
            try:
                with open(f) as fp:
                    data = yaml.safe_load(fp)
                if data.get("id") == name:
                    f.unlink()
                    return True
            except Exception:
                continue
        
        return False
    
    async def exists(self, name: str) -> bool:
        """Check if a recording exists."""
        file_path = self._get_file_path(name)
        if file_path.exists():
            return True
        
        # Check by ID
        for f in self.directory.glob("*.yaml"):
            try:
                with open(f) as fp:
                    data = yaml.safe_load(fp)
                if data.get("id") == name:
                    return True
            except Exception:
                continue
        
        return False
