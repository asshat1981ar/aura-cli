import pytest
import uvicorn
import threading
import time
from fastapi import FastAPI, Request
from typing import Dict, Any

class MockMCPServer:
    def __init__(self, port: int):
        self.port = port
        self.app = FastAPI()
        self.thread = None
        self._setup_routes()

    def _setup_routes(self):
        @self.app.get("/health")
        async def health():
            return {"status": "ok", "version": "1.0.0"}

        @self.app.get("/tools")
        async def tools():
            return {"tools": [{"name": "echo", "description": "echo back"}]}

        @self.app.post("/call")
        async def call(req: Dict[str, Any]):
            name = req.get("name")
            args = req.get("arguments", {})
            return {"status": "success", "result": f"Echo: {args.get('text', '')}"}

    def start(self):
        self.thread = threading.Thread(
            target=uvicorn.run,
            args=(self.app,),
            kwargs={"host": "127.0.0.1", "port": self.port, "log_level": "error"},
            daemon=True
        )
        self.thread.start()
        # Give it a moment to start
        time.sleep(1)

@pytest.fixture
def mock_mcp_server():
    server = MockMCPServer(port=9001)
    server.start()
    return server
