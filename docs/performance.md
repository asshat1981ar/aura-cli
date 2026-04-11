# AURA CLI Performance Guide

This document outlines performance targets, current benchmarks, and optimization strategies for AURA CLI.

## Performance Targets

| Metric | Target | Current Status |
|--------|--------|----------------|
| Cold start (--version) | <300ms | ✅ ~188ms (median) |
| Import core modules | <200ms | ✅ ~1.2ms |
| Import aura_cli.cli_options | <100ms | ⚠️ ~206ms |
| Import aura_cli.dispatch | <500ms | ❌ ~2600ms |
| pytest collection | <5s | ⏳ TBD |

## Current Benchmarks

### Startup Time Measurements

```bash
$ python scripts/benchmark_startup.py
```

Typical results on a modern development machine:

```
📊 Command: version
----------------------------------------
  Iterations: 5
  Errors:     0
  Min:        152.5 ms
  Max:        227.1 ms
  Mean:       188.0 ms
  Median:     185.0 ms
  Target:     300 ms ✅ PASS
```

### Import Time Analysis

```bash
$ python scripts/profile_imports.py --report
```

Key findings:
- `aura_cli.dispatch` is the heaviest import (~2600ms)
- `aura_cli.options` takes ~60ms
- Heavy third-party dependencies include various ML/NLP libraries

## Optimization Strategies

### 1. Lazy Loading

Use the `core.lazy_imports` module to defer expensive imports:

```python
# ❌ BAD: Import at module load time
from memory.brain import Brain
from core.orchestrator import LoopOrchestrator

# ✅ GOOD: Lazy import - module loaded only when used
from core.lazy_imports import LazyBrain, LazyLoopOrchestrator

brain = LazyBrain()  # memory.brain imported here
orchestrator = LazyLoopOrchestrator()  # core.orchestrator imported here
```

### 2. Lazy Import Utilities

The `core.lazy_imports` module provides several utilities:

#### LazyImport Class

```python
from core.lazy_imports import LazyImport

# Lazy class import
Brain = LazyImport("memory.brain", "Brain")
brain_instance = Brain()  # Imports on first use
```

#### LazyModule Class

```python
from core.lazy_imports import LazyModule

numpy = LazyModule("numpy")
arr = numpy.array([1, 2, 3])  # Imports numpy here
```

#### Pre-defined Lazy Imports

For common AURA components:

```python
from core.lazy_imports import (
    LazyBrain,
    LazyLoopOrchestrator,
    LazyDebuggerAgent,
    LazyPlannerAgent,
    LazyConfigManager,
    # ... and more
)
```

### 3. Module-level Optimizations

#### Package `__init__.py` Files

Keep `__init__.py` files minimal. Avoid:

```python
# ❌ BAD: Heavy imports in __init__.py
from .heavy_module import HeavyClass  # Slows down all imports

# ✅ GOOD: Minimal __init__.py
"""Package docstring."""
__version__ = "1.0.0"
```

#### CLI Entry Point

The `main.py` entry point uses a shim pattern to keep lightweight paths fast:

```python
# main.py - Only imports needed for help/version
from aura_cli.cli_options import parse_cli_args, render_help

# Heavy imports deferred until actually needed
if needs_full_runtime:
    from aura_cli.cli_main import main as _main
```

## Profiling Tools

### Benchmark Script

```bash
# Basic benchmark
python scripts/benchmark_startup.py

# More iterations
python scripts/benchmark_startup.py --iterations 10

# Custom commands
python scripts/benchmark_startup.py --commands version,help,mcp-tools

# JSON output
python scripts/benchmark_startup.py --json > benchmarks.json
```

### Import Profiler

```bash
# Profile single module
python scripts/profile_imports.py --module aura_cli.cli_main

# Show top 30 imports
python scripts/profile_imports.py --top 30

# Comprehensive report
python scripts/profile_imports.py --report

# JSON output
python scripts/profile_imports.py --report --json > import_report.json
```

## Performance Checklist

When adding new features:

- [ ] Keep `__init__.py` files minimal
- [ ] Use lazy imports for heavy dependencies (ML libs, DB connections)
- [ ] Benchmark startup time with `scripts/benchmark_startup.py`
- [ ] Profile imports with `scripts/profile_imports.py`
- [ ] Ensure `--version` remains under 300ms
- [ ] Document any new heavy dependencies

## Known Bottlenecks

### High Priority

1. **aura_cli.dispatch** (~2600ms)
   - Imports heavy orchestration components
   - Recommendation: Use lazy imports for dispatch handlers

### Medium Priority

2. **aura_cli.cli_options** (~206ms)
   - Command spec processing
   - Recommendation: Cache compiled specs

### Low Priority

3. **Third-party ML libraries**
   - PyTorch, Transformers, etc.
   - Recommendation: Always use lazy imports

## Future Improvements

Planned optimizations:

1. **Import caching**: Cache compiled import graphs
2. **Deferred loading**: Load heavy agents only when needed
3. **Module splitting**: Split CLI into lighter sub-packages
4. **Startup parallelization**: Parallelize independent imports

## Measuring Performance

Always measure before and after optimizations:

```bash
# Baseline measurement
python scripts/benchmark_startup.py --iterations 10 > baseline.txt

# Make optimizations...

# Verify improvement
python scripts/benchmark_startup.py --iterations 10 > optimized.txt

diff baseline.txt optimized.txt
```

## CI Integration

Add to CI pipeline to prevent regressions:

```yaml
# .github/workflows/performance.yml
- name: Benchmark startup
  run: |
    python scripts/benchmark_startup.py --json > startup.json
    # Fail if median > 300ms
    jq '.results[] | select(.command == "version" and .median_ms > 300)' startup.json | grep -q . && exit 1
```

## References

- [Python Import System](https://docs.python.org/3/reference/import.html)
- [Importtime trace](https://docs.python.org/3/using/cmdline.html#cmdoption-X)
- AURA Lazy Imports: `core/lazy_imports.py`
