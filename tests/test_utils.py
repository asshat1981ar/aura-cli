"""Unit tests for core/utils.py — FileUtils, ConfigLoader, ErrorHandler, Logger."""

import json
import os
import tempfile
from pathlib import Path

import pytest

from core.utils import ConfigLoader, FileUtils


class TestFileUtilsSaveLoadRoundTrip:
    def test_save_and_load_json_roundtrip(self):
        data = {"key": "value", "number": 42, "nested": {"a": [1, 2, 3]}}
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            FileUtils.save_json(data, path)
            loaded = FileUtils.load_json(path)
            assert loaded == data
        finally:
            os.unlink(path)

    def test_save_json_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "output.json")
            FileUtils.save_json({"x": 1}, path)
            assert os.path.exists(path)

    def test_save_json_uses_indent(self):
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            path = f.name
        try:
            FileUtils.save_json({"a": 1}, path, indent=4)
            content = Path(path).read_text()
            assert "    " in content  # 4-space indent present
        finally:
            os.unlink(path)

    def test_load_json_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            FileUtils.load_json("/tmp/does_not_exist_aura_test.json")

    def test_save_json_empty_dict(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            FileUtils.save_json({}, path)
            assert FileUtils.load_json(path) == {}
        finally:
            os.unlink(path)


class TestFileUtilsEnsurePathExists:
    def test_creates_parent_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            deep_path = os.path.join(tmpdir, "a", "b", "c", "file.json")
            FileUtils.ensure_path_exists(deep_path)
            assert Path(deep_path).parent.exists()

    def test_does_not_create_the_file_itself(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "subdir", "data.json")
            FileUtils.ensure_path_exists(file_path)
            assert not os.path.exists(file_path)

    def test_idempotent_on_existing_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "file.json")
            FileUtils.ensure_path_exists(file_path)
            FileUtils.ensure_path_exists(file_path)  # Should not raise


class TestConfigLoader:
    def test_load_config_from_file(self):
        settings = {"model": "gemini", "max_cycles": 10}
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump(settings, f)
            path = f.name
        try:
            loader = ConfigLoader(path)
            assert loader.get("model") == "gemini"
            assert loader.get("max_cycles") == 10
        finally:
            os.unlink(path)

    def test_get_returns_default_for_missing_key(self):
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump({}, f)
            path = f.name
        try:
            loader = ConfigLoader(path)
            assert loader.get("missing_key", "fallback") == "fallback"
        finally:
            os.unlink(path)

    def test_missing_config_file_returns_empty_settings(self):
        loader = ConfigLoader("/tmp/no_such_config_aura_test.json")
        assert loader.settings == {}
        assert loader.get("anything", 99) == 99
