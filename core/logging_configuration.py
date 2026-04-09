import logging


def setup_logging(log_level=logging.INFO):
    # logging.basicConfig is intentionally NOT called here —
    # AURA uses log_json (core.logging_utils) for structured JSON logging.
    # Standard library loggers from third-party packages are silenced at WARNING
    # to prevent noisy output interfering with structured logs.
    logging.root.setLevel(log_level)
    if not logging.root.handlers:
        # Minimal handler so stdlib loggers can still emit to stderr if needed,
        # but format is kept consistent with structured output.
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))
        logging.root.addHandler(handler)
