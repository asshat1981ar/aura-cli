#!/usr/bin/env python3
import argparse
import json
import socket
import sys
import time
import urllib.error
import urllib.request
import urllib.parse
from pathlib import Path


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


def _load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("Config root must be a JSON object.")
    return data


def _profile_name_for_route(config_data: dict, route_key: str) -> str | None:
    local_routing = config_data.get("local_model_routing", {}) or {}
    value = local_routing.get(route_key)
    if isinstance(value, str) and value:
        return value
    return None


def _embedding_profile_name(config_data: dict) -> str | None:
    semantic_memory = config_data.get("semantic_memory", {}) or {}
    embedding_model = semantic_memory.get("embedding_model")
    if isinstance(embedding_model, str) and embedding_model.startswith("local_profile:"):
        return embedding_model.split(":", 1)[1]
    return _profile_name_for_route(config_data, "embedding")


def _profile_target(profile: dict) -> tuple[str, int] | None:
    if not isinstance(profile, dict):
        return None
    provider = str(profile.get("provider", "")).strip().lower()
    if provider not in {"openai_compatible", "ollama"}:
        return None

    base_url = profile.get("base_url")
    if not isinstance(base_url, str) or not base_url:
        return None

    parsed = urllib.parse.urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or not parsed.port:
        return None
    return parsed.hostname, parsed.port


def build_checks_from_config(config_data: dict) -> list[tuple[str, str, int]]:
    profiles = config_data.get("local_model_profiles", {}) or {}
    if not isinstance(profiles, dict):
        return []

    checks: list[tuple[str, str, int]] = []
    seen_targets: set[tuple[str, int]] = set()
    ordered_profiles = [
        ("coder", _profile_name_for_route(config_data, "code_generation")),
        ("planner", _profile_name_for_route(config_data, "planning")),
        ("embedding", _embedding_profile_name(config_data)),
    ]

    for label, profile_name in ordered_profiles:
        if not isinstance(profile_name, str) or not profile_name:
            continue
        target = _profile_target(profiles.get(profile_name, {}))
        if target is None:
            continue
        if target in seen_targets:
            continue
        seen_targets.add(target)
        checks.append((label, target[0], target[1]))

    return checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check that the Android local coder, planner, and optional embedding model servers are reachable.")
    parser.add_argument(
        "--config",
        default=None,
        help="Optional AURA config JSON path. When set, derive coder/planner/embedding endpoints from local_model_profiles.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--coder-port", type=int, default=8080)
    parser.add_argument("--planner-port", type=int, default=8081)
    parser.add_argument("--embedding-port", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=20.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.config:
        try:
            checks = build_checks_from_config(_load_config(args.config))
        except Exception as exc:
            print(f"failed to load config: {exc}", file=sys.stderr)
            return 2
        if not checks:
            print("no HTTP-backed local model endpoints found in config", file=sys.stderr)
            return 2
    else:
        checks = [
            ("coder", args.host, args.coder_port),
            ("planner", args.host, args.planner_port),
        ]
        if args.embedding_port is not None:
            checks.append(("embedding", args.host, args.embedding_port))

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
