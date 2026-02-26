import base64
from fastapi.testclient import TestClient
from tools import mcp_server as server


def setup_auth(monkeypatch):
    monkeypatch.setenv("MCP_API_TOKEN", "t")
    return TestClient(server.app), {"Authorization": "Bearer t"}


def test_compress_and_decompress_skips(monkeypatch, tmp_path):
    c, hdrs = setup_auth(monkeypatch)
    monkeypatch.setattr(server, "ENABLE_WRITE", True)
    small = server.PROJECT_ROOT / "tmp_small.txt"
    big = server.PROJECT_ROOT / "tmp_big.txt"
    small.write_text("ok")
    big.write_text("x" * (server.COMPRESS_MAX_BYTES + 10))
    resp = c.post(
        "/call",
        headers=hdrs,
        json={"tool_name": "compress", "args": {"paths": [str(small.relative_to(server.PROJECT_ROOT)), str(big.relative_to(server.PROJECT_ROOT))]}},
    )
    small.unlink(missing_ok=True)
    big.unlink(missing_ok=True)
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert "tmp_big.txt" in data["skipped"]
    b64 = data["base64"]
    # decompress back
    resp2 = c.post(
        "/call",
        headers=hdrs,
        json={"tool_name": "decompress", "args": {"base64": b64, "dest": "tmp_out"}},
    )
    assert resp2.status_code == 200
    assert resp2.json()["data"]["skipped"] == []


def test_tail_logs_binary_guard(monkeypatch, tmp_path):
    c, hdrs = setup_auth(monkeypatch)
    log = server.PROJECT_ROOT / "tmp_bin.log"
    log.write_bytes(b"\x00\x01\x02")
    r = c.post("/call", headers=hdrs, json={"tool_name": "tail_logs", "args": {"path": str(log.relative_to(server.PROJECT_ROOT))}})
    log.unlink(missing_ok=True)
    assert r.status_code == 200
    lines = r.json()["data"]["lines"]
    assert lines == ["[binary content skipped]"]
