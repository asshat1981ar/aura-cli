"""Recording session manager."""

from typing import Dict, List, Optional

from .models import Recording, RecordingStep, ReplayResult
from .storage import RecordingStorage


class RecordingSession:
    """Context manager for recording a session."""
    
    def __init__(self, name: str, storage: Optional[RecordingStorage] = None):
        self.recording = Recording(name=name)
        self.storage = storage or RecordingStorage()
        self._active = False
    
    def __enter__(self):
        """Start recording session."""
        self._active = True
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End recording session and save."""
        self._active = False
        return False
    
    def record_step(
        self,
        command: str,
        *args,
        condition: Optional[str] = None,
        retry_count: int = 3,
        retry_delay: float = 1.0,
        timeout: int = 60,
        **kwargs,
    ) -> RecordingStep:
        """Record a step."""
        if not self._active:
            raise RuntimeError("Recording session not active")
        
        step = RecordingStep(
            command=command,
            args=list(args),
            kwargs=kwargs,
            condition=condition,
            retry_count=retry_count,
            retry_delay=retry_delay,
            timeout=timeout,
        )
        self.recording.add_step(step)
        return step
    
    def set_variable(self, key: str, value: str):
        """Set a variable for the recording."""
        self.recording.variables[key] = value
    
    async def save(self) -> str:
        """Save the recording."""
        path = await self.storage.save(self.recording)
        return str(path)


class Recorder:
    """Main recorder API."""
    
    def __init__(self, storage: Optional[RecordingStorage] = None):
        self.storage = storage or RecordingStorage()
    
    def start_recording(self, name: str) -> RecordingSession:
        """Start a new recording session."""
        return RecordingSession(name, self.storage)
    
    async def list_recordings(self) -> List[Recording]:
        """List all recordings."""
        return await self.storage.list()
    
    async def load(self, name: str) -> Optional[Recording]:
        """Load a recording by name."""
        return await self.storage.load(name)
    
    async def delete(self, name: str) -> bool:
        """Delete a recording."""
        return await self.storage.delete(name)
    
    async def exists(self, name: str) -> bool:
        """Check if a recording exists."""
        return await self.storage.exists(name)
