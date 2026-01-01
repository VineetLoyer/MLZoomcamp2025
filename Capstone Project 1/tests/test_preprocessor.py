"""
Unit tests for TextPreprocessor.

Tests edge cases including empty strings, unicode characters, and very long text.
Requirements: 1.2, 1.3
"""

import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocessor import TextPreprocessor


class TestTextPreprocessorCleanText:
    """Tests for the clean_text method."""
    
    @pytest.fixture
    def preprocessor(self):
        return TextPreprocessor()
    
    # --- Empty and None input tests ---
    
    def test_empty_string(self, preprocessor):
        """Empty string should return empty string."""
        assert preprocessor.clean_text("") == ""
    
    def test_none_input(self, preprocessor):
        """None input should return empty string."""
        assert preprocessor.clean_text(None) == ""
    
    def test_whitespace_only(self, preprocessor):
        """Whitespace-only string should return empty string."""
        assert preprocessor.clean_text("   \t\n  ") == ""
    
    # --- URL removal tests ---
    
    def test_removes_http_url(self, preprocessor):
        """HTTP URLs should be removed."""
        text = "Check this http://example.com link"
        result = preprocessor.clean_text(text)
        assert "http" not in result
        assert "example.com" not in result
    
    def test_removes_https_url(self, preprocessor):
        """HTTPS URLs should be removed."""
        text = "Visit https://www.example.com/path?query=1"
        result = preprocessor.clean_text(text)
        assert "https" not in result
        assert "example.com" not in result
    
    # --- MBTI type removal tests ---
    
    def test_removes_mbti_types(self, preprocessor):
        """MBTI type mentions should be removed."""
        text = "I am an INTJ and my friend is ENFP"
        result = preprocessor.clean_text(text)
        assert "intj" not in result
        assert "enfp" not in result
    
    def test_removes_lowercase_mbti(self, preprocessor):
        """Lowercase MBTI types should also be removed."""
        text = "Some people say intj types are analytical"
        result = preprocessor.clean_text(text)
        assert "intj" not in result
    
    # --- Lowercase conversion tests ---
    
    def test_converts_to_lowercase(self, preprocessor):
        """Text should be converted to lowercase."""
        text = "HELLO World"
        result = preprocessor.clean_text(text)
        assert result == "hello world"
    
    # --- Whitespace normalization tests ---
    
    def test_normalizes_whitespace(self, preprocessor):
        """Multiple whitespace should be normalized to single space."""
        text = "hello    world   test"
        result = preprocessor.clean_text(text)
        assert "  " not in result
        assert result == "hello world test"
    
    def test_removes_newlines_and_tabs(self, preprocessor):
        """Newlines and tabs should be normalized."""
        text = "hello\nworld\ttest"
        result = preprocessor.clean_text(text)
        assert "\n" not in result
        assert "\t" not in result
    
    # --- Unicode tests ---
    
    def test_handles_unicode_characters(self, preprocessor):
        """Unicode characters should be handled (special chars removed)."""
        text = "Hello 世界 café résumé"
        result = preprocessor.clean_text(text)
        # Should still produce valid output
        assert isinstance(result, str)
        assert "hello" in result
    
    def test_handles_emojis(self, preprocessor):
        """Emojis should be removed as special characters."""
        text = "I love coding 😀 🎉"
        result = preprocessor.clean_text(text)
        assert "😀" not in result
        assert "🎉" not in result
        assert "love coding" in result
    
    # --- Very long text tests ---
    
    def test_handles_very_long_text(self, preprocessor):
        """Very long text should be processed without error."""
        text = "word " * 10000  # 50000+ characters
        result = preprocessor.clean_text(text)
        assert isinstance(result, str)
        assert len(result) > 0
    
    # --- Special characters tests ---
    
    def test_preserves_basic_punctuation(self, preprocessor):
        """Basic punctuation should be preserved."""
        text = "Hello, world! How are you?"
        result = preprocessor.clean_text(text)
        assert "," in result
        assert "!" in result
        assert "?" in result


class TestTextPreprocessorPreprocessPosts:
    """Tests for the preprocess_posts method."""
    
    @pytest.fixture
    def preprocessor(self):
        return TextPreprocessor()
    
    def test_empty_posts(self, preprocessor):
        """Empty posts string should return empty string."""
        assert preprocessor.preprocess_posts("") == ""
    
    def test_none_posts(self, preprocessor):
        """None input should return empty string."""
        assert preprocessor.preprocess_posts(None) == ""
    
    def test_splits_by_delimiter(self, preprocessor):
        """Posts should be split by ||| delimiter."""
        posts = "First post|||Second post|||Third post"
        result = preprocessor.preprocess_posts(posts)
        assert "first post" in result
        assert "second post" in result
        assert "third post" in result
        assert "|||" not in result
    
    def test_cleans_each_post(self, preprocessor):
        """Each post should be cleaned individually."""
        posts = "Check http://example.com|||I am INTJ|||HELLO WORLD"
        result = preprocessor.preprocess_posts(posts)
        assert "http" not in result
        assert "intj" not in result
        assert result.islower()
    
    def test_filters_empty_posts(self, preprocessor):
        """Empty posts after cleaning should be filtered out."""
        posts = "Hello|||   |||World"
        result = preprocessor.preprocess_posts(posts)
        # Should not have extra spaces from empty middle post
        assert "  " not in result
    
    def test_custom_delimiter(self, preprocessor):
        """Custom delimiter should work."""
        posts = "First post###Second post###Third post"
        result = preprocessor.preprocess_posts(posts, delimiter="###")
        assert "first post" in result
        assert "second post" in result
    
    def test_single_post_no_delimiter(self, preprocessor):
        """Single post without delimiter should work."""
        posts = "Just a single post here"
        result = preprocessor.preprocess_posts(posts)
        assert result == "just a single post here"


class TestTextPreprocessorPreprocessBatch:
    """Tests for the preprocess_batch method."""
    
    @pytest.fixture
    def preprocessor(self):
        return TextPreprocessor()
    
    def test_empty_batch(self, preprocessor):
        """Empty batch should return empty list."""
        assert preprocessor.preprocess_batch([]) == []
    
    def test_batch_processing(self, preprocessor):
        """Batch of texts should all be processed."""
        texts = [
            "First text|||with posts",
            "Second TEXT|||MORE POSTS",
            "Third http://url.com|||INTJ type"
        ]
        results = preprocessor.preprocess_batch(texts)
        assert len(results) == 3
        assert all(isinstance(r, str) for r in results)
        assert all(r.islower() or r == "" for r in results)
