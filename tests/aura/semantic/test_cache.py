"""Tests for analysis cache."""

import pytest
import time

from aura.semantic.cache import AnalysisCache


class TestAnalysisCache:
    @pytest.fixture
    def cache(self, tmp_path):
        db_path = tmp_path / "test_cache.db"
        return AnalysisCache(db_path=str(db_path), ttl=60)
    
    def test_cache_set_and_get(self, cache):
        result = {"elements": [{"name": "func1"}], "total": 1}
        
        cache.set("/path/to/file.py", "def func1(): pass", result)
        cached = cache.get("/path/to/file.py", "def func1(): pass")
        
        assert cached == result
    
    def test_cache_miss_different_content(self, cache):
        result = {"elements": [{"name": "func1"}], "total": 1}
        
        cache.set("/path/to/file.py", "def func1(): pass", result)
        cached = cache.get("/path/to/file.py", "def func2(): pass")
        
        assert cached is None
    
    def test_cache_expiration(self, tmp_path):
        db_path = tmp_path / "test_cache.db"
        cache = AnalysisCache(db_path=str(db_path), ttl=0)  # Immediate expiration
        
        result = {"elements": [{"name": "func1"}]}
        cache.set("/path/to/file.py", "def func1(): pass", result)
        
        # Should be expired immediately
        cached = cache.get("/path/to/file.py", "def func1(): pass")
        assert cached is None
    
    def test_invalidate_by_file(self, cache):
        result1 = {"elements": [{"name": "func1"}]}
        result2 = {"elements": [{"name": "func2"}]}
        
        cache.set("/path/file1.py", "content1", result1)
        cache.set("/path/file2.py", "content2", result2)
        
        cache.invalidate("/path/file1.py")
        
        assert cache.get("/path/file1.py", "content1") is None
        assert cache.get("/path/file2.py", "content2") == result2
    
    def test_invalidate_all(self, cache):
        result1 = {"elements": [{"name": "func1"}]}
        result2 = {"elements": [{"name": "func2"}]}
        
        cache.set("/path/file1.py", "content1", result1)
        cache.set("/path/file2.py", "content2", result2)
        
        cache.invalidate()
        
        assert cache.get("/path/file1.py", "content1") is None
        assert cache.get("/path/file2.py", "content2") is None
    
    def test_clear_expired(self, cache):
        # Add entry with short TTL
        cache.set("/path/file.py", "content", {"data": "value"}, ttl=1)
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Clear expired entries
        cache.clear_expired()
        
        # Should be gone
        assert cache.get("/path/file.py", "content") is None
    
    def test_get_stats(self, cache):
        cache.set("/path/file1.py", "content1", {"data": "1"})
        cache.set("/path/file2.py", "content2", {"data": "2"})
        
        stats = cache.get_stats()
        
        assert stats["total_entries"] == 2
        assert stats["valid_entries"] == 2
        assert stats["expired_entries"] == 0
    
    def test_cache_update(self, cache):
        result1 = {"elements": [{"name": "func1"}]}
        result2 = {"elements": [{"name": "func1_updated"}]}
        
        cache.set("/path/file.py", "content", result1)
        cache.set("/path/file.py", "content", result2)
        
        cached = cache.get("/path/file.py", "content")
        
        assert cached == result2
