.PHONY: fuzz test lint

fuzz:
	pytest tests/security/fuzz_*.py -v --hypothesis-seed=random \
	  -k "fuzz or Fuzz" \
	  --timeout=300

test:
	pytest tests/ -v --timeout=120

lint:
	ruff check .
