"""CLI commands for running the AURA API server."""

import typer
import uvicorn

from core.logging_utils import log_json

app = typer.Typer(help="AURA API server commands")


@app.command()
def run(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind to"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload"),
    workers: int = typer.Option(1, "--workers", "-w", help="Number of worker processes"),
):
    """Run the AURA API server."""
    log_json("INFO", "api_server_starting", {"host": host, "port": port})
    
    uvicorn.run(
        "aura_cli.api_server:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers if not reload else 1,
        log_level="info",
    )


@app.command()
def status():
    """Check API server status."""
    import requests
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            typer.echo("✓ API server is running")
            typer.echo(f"  Status: {data['status']}")
            typer.echo(f"  Timestamp: {data['timestamp']}")
        else:
            typer.echo(f"✗ API server returned status {response.status_code}")
    except requests.ConnectionError:
        typer.echo("✗ API server is not running")
    except Exception as e:
        typer.echo(f"✗ Error checking API status: {e}")


if __name__ == "__main__":
    app()
