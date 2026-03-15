#!/usr/bin/env python3
import requests
import sys
import json

def test_server(name, port):
    base_url = f"http://localhost:{port}"
    print(f"Testing {name} at {base_url}...")
    
    # 1. Health check
    try:
        resp = requests.get(f"{base_url}/health", timeout=2)
        resp.raise_for_status()
        print(f"  [PASS] Health check: {resp.json()}")
    except Exception as e:
        print(f"  [FAIL] Health check: {e}")
        return False

    # 2. List tools
    try:
        resp = requests.get(f"{base_url}/tools", timeout=2)
        resp.raise_for_status()
        
        data = resp.json()
        tools = []
        if isinstance(data, dict):
            tools = data.get("data", {}).get("tools", [])
            if not tools:
                 tools = data.get("tools", [])
        elif isinstance(data, list):
             tools = data
             
        print(f"  [PASS] List tools: Found {len(tools)} tools")
    except Exception as e:
        print(f"  [FAIL] List tools: {e}")
        return False
        
    return True

def test_dev_tools_execution():
    print("\nTesting Dev Tools execution (read_file)...")
    url = "http://localhost:8001/call"
    payload = {
        "tool_name": "read_file",
        "args": {
            "path": "aura.config.json"
        }
    }
    try:
        resp = requests.post(url, json=payload, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        content = data.get("data", {}).get("content", "")
        if "mcp_servers" in content:
            print("  [PASS] read_file returned valid content")
        else:
            print(f"  [FAIL] read_file returned unexpected content: {content[:100]}...")
    except Exception as e:
        print(f"  [FAIL] Execution failed: {e}")

if __name__ == "__main__":
    print("=== Verifying AURA MCP Servers ===\n")
    
    servers = [
        ("Dev Tools", 8001),
        ("Skills", 8002),
        ("Control", 8003),
        ("Agentic Loop", 8006),
        ("Copilot", 8007)
    ]
    
    results = []
    for name, port in servers:
        results.append(test_server(name, port))
        
    if all(results):
        test_dev_tools_execution()
        print("\n=== All MCP Servers Verified! ===")
        sys.exit(0)
    else:
        print("\n=== Some servers failed verification ===")
        sys.exit(1)
