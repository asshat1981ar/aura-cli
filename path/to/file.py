import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

def setup_logging():
    logging.basicConfig(level=logging.INFO)

@dataclass
class SomeDataClass:
    field1: str
    field2: int

# Other functions and classes can use setup_logging and logger
