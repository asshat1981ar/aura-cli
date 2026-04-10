#!/usr/bin/env python3
"""Export FastAPI OpenAPI spec to docs/api/openapi.json."""
import json
import sys
from pathlib import Path


def export_openapi():
    try:
        # Import the FastAPI app
        import os

        os.environ.setdefault("AURA_TEST_MODE", "1")
        from aura_cli.server import app

        spec = app.openapi()
        out = Path("docs/api")
        out.mkdir(parents=True, exist_ok=True)
        out_file = out / "openapi.json"
        out_file.write_text(json.dumps(spec, indent=2))
        print(f"OpenAPI spec written to {out_file} ({len(spec.get('paths', {}))} paths)")
        return 0
    except Exception as e:
        print(f"Error exporting OpenAPI spec: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(export_openapi())
