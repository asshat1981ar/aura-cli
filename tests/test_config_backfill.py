"""
Tests for backfill configuration in ConfigManager.
"""
from __future__ import annotations

import pytest
import os
from pathlib import Path
from unittest.mock import patch
from core.config_manager import ConfigManager

def test_config_manager_has_backfill_defaults():
    """
    Verify that the new settings have sensible defaults.
    """
    cm = ConfigManager(config_file="nonexistent.json")
    # These should be in DEFAULT_CONFIG once implemented
    assert cm.get("auto_backfill_coverage") is True
    assert cm.get("reliability_threshold") == 80.0

def test_config_manager_validates_reliability_threshold():
    """
    Verify that reliability_threshold is constrained to [0, 100].
    """
    cm = ConfigManager()
    
    # Valid
    cm.set_runtime_override("reliability_threshold", 50.5)
    assert cm.get("reliability_threshold") == 50.5
    
    # Invalid (too high) - should fall back to default
    cm.set_runtime_override("reliability_threshold", 150)
    assert cm.get("reliability_threshold") == 80.0
    
    # Invalid (type)
    cm.set_runtime_override("reliability_threshold", "not-a-number")
    assert cm.get("reliability_threshold") == 80.0

def test_config_manager_handles_env_overrides():
    with patch.dict(os.environ, {"AURA_AUTO_BACKFILL_COVERAGE": "false", "AURA_RELIABILITY_THRESHOLD": "95"}):
        cm = ConfigManager()
        assert cm.get("auto_backfill_coverage") is False
        assert cm.get("reliability_threshold") == 95.0
