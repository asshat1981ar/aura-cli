# Refactored for clarity and performance
import sys
import os


def setup():
    """Initialize the AURA CLI environment."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


setup()
