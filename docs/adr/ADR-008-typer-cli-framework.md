# ADR-008: Typer CLI Framework Choice

**Date:** 2026-04-10  
**Status:** Accepted  
**Deciders:** AURA Core Team  

## Context

AURA CLI needed a robust command-line interface framework that would:

1. Provide excellent developer experience for building complex CLI applications
2. Support type hints and automatic validation
3. Generate helpful help messages automatically
4. Support nested subcommands (e.g., `aura goal add`, `aura mcp tools`)
5. Integrate well with modern Python tooling
6. Allow for automatic completion generation
7. Support both sync and async command handlers

The main contenders were:
- **argparse** (stdlib) — Built-in, no dependencies, but verbose and limited
- **Click** — Popular, battle-tested, but doesn't leverage Python type hints
- **Typer** — Built on Click, adds type hint support and modern DX
- **argparse + custom** — Maximum control but high maintenance burden

## Decision

We chose **Typer** as our CLI framework.

### Rationale

1. **Type Safety**: Typer leverages Python type hints for automatic validation and conversion
   ```python
   # Typer automatically validates and converts types
   def command(name: str, count: int = 1, verbose: bool = False):
       ...
   ```

2. **Developer Experience**: Minimal boilerplate for complex CLI structures
   ```python
   import typer
   
   app = typer.Typer()
   
   @app.command()
   def hello(name: str):
       typer.echo(f"Hello {name}")
   
   if __name__ == "__main__":
       app()
   ```

3. **Automatic Help Generation**: Rich help text from docstrings and type hints

4. **Nested Commands**: Natural support for complex command hierarchies
   ```python
   app = typer.Typer()
   goal_app = typer.Typer()
   app.add_typer(goal_app, name="goal")
   
   @goal_app.command("add")
   def goal_add(description: str):
       ...
   ```

5. **Shell Completion**: Automatic bash/zsh/fish completion generation

6. **Rich Integration**: Seamless integration with the Rich library for beautiful output

7. **FastAPI Heritage**: Same author as FastAPI, consistent design philosophy

## Consequences

### Positive

- Reduced boilerplate code in CLI implementation
- Type safety catches errors at development time
- Automatic validation reduces runtime errors
- Self-documenting CLI through type hints and docstrings
- Easy to extend with new commands
- Consistent patterns across the CLI surface

### Negative

- Additional dependency (though Typer is lightweight)
- Learning curve for developers unfamiliar with type hints
- Less control over argument parsing compared to argparse
- Some advanced Click features require deeper understanding

### Migration Path

- Legacy Click-based code in `aura_cli/` was migrated incrementally
- All new commands use Typer exclusively
- Custom middleware layer (`aura_cli/middleware.py`) provides additional abstraction

## References

- [Typer Documentation](https://typer.tiangolo.com/)
- [CLI Implementation](https://github.com/asshat1981ar/aura-cli/tree/main/aura_cli)
- [Example Commands](https://github.com/asshat1981ar/aura-cli/tree/main/aura_cli/commands.py)
