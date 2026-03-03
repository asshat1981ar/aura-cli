#!/usr/bin/env python3
import argparse
import socket
import sys
import time
import urllib.error
import urllib.request


def probe_http(host: str, port: int, path: str, timeout: float) -> tuple[bool, str]:
    url = f"http://{host}:{port}{path}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return True, f"{url} -> HTTP {response.status}"
    except urllib.error.HTTPError as exc:
        if exc.code < 500:
            return True, f"{url} -> HTTP {exc.code}"
        return False, f"{url} -> HTTP {exc.code}"
    except Exception as exc:
        return False, f"{url} -> {exc}"


def probe_port(host: str, port: int, timeout: float) -> tuple[bool, str]:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True, f"tcp://{host}:{port} reachable"
    except OSError as exc:
        return False, f"tcp://{host}:{port} unreachable: {exc}"


def check_model_endpoint(host: str, port: int, timeout: float) -> tuple[bool, str]:
    tcp_ok, tcp_message = probe_port(host, port, timeout)
    if not tcp_ok:
        return False, tcp_message

    for path in ("/health", "/v1/models", "/"):
        ok, message = probe_http(host, port, path, timeout)
        if ok:
            return True, message

    return False, tcp_message


def wait_for_model_endpoint(name: str, host: str, port: int, timeout: float) -> tuple[bool, str]:
    deadline = time.time() + timeout
    last_message = f"{name}: waiting"

    while time.time() < deadline:
        ok, message = check_model_endpoint(host, port, min(timeout, 2.0))
        last_message = f"{name}: {message}"
        if ok:
            return True, last_message
        time.sleep(0.5)

    return False, last_message


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check that the Android local coder and planner model servers are reachable."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--coder-port", type=int, default=8080)
    parser.add_argument("--planner-port", type=int, default=8081)
    parser.add_argument("--timeout", type=float, default=20.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checks = [
        ("coder", args.host, args.coder_port),
        ("planner", args.host, args.planner_port),
    ]

    all_ok = True
    for name, host, port in checks:
        ok, message = wait_for_model_endpoint(name, host, port, args.timeout)
        print(message)
        if not ok:
            all_ok = False

    if all_ok:
        print("android local models ready")
        return 0

    print("android local models not ready", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
