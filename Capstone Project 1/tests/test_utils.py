"""
Unit tests for MBTI label encoding utilities.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils import (
    encode_mbti,
    decode_mbti,
    encode_mbti_list,
    decode_mbti_list,
    VALID_MBTI_TYPES
)


class TestEncodeMbti:
    """Tests for encode_mbti function."""
    
    def test_encode_intj(self):
        """INTJ should encode to all zeros (I=0, N=0, T=0, J=0)."""
        result = encode_mbti('INTJ')
        assert result == {'IE': 0, 'NS': 0, 'TF': 0, 'JP': 0}
    
    def test_encode_enfp(self):
        """ENFP should encode to E=1, N=0, F=1, P=1."""
        result = encode_mbti('ENFP')
        assert result == {'IE': 1, 'NS': 0, 'TF': 1, 'JP': 1}
    
    def test_encode_esfp(self):
        """ESFP should encode to all ones (E=1, S=1, F=1, P=1)."""
        result = encode_mbti('ESFP')
        assert result == {'IE': 1, 'NS': 1, 'TF': 1, 'JP': 1}
    
    def test_encode_lowercase(self):
        """Should handle lowercase input."""
        result = encode_mbti('intj')
        assert result == {'IE': 0, 'NS': 0, 'TF': 0, 'JP': 0}
    
    def test_encode_mixed_case(self):
        """Should handle mixed case input."""
        result = encode_mbti('InTj')
        assert result == {'IE': 0, 'NS': 0, 'TF': 0, 'JP': 0}
    
    def test_encode_with_whitespace(self):
        """Should handle input with leading/trailing whitespace."""
        result = encode_mbti('  INTJ  ')
        assert result == {'IE': 0, 'NS': 0, 'TF': 0, 'JP': 0}
    
    def test_encode_invalid_type(self):
        """Should raise ValueError for invalid MBTI type."""
        with pytest.raises(ValueError) as exc_info:
            encode_mbti('XXXX')
        assert 'Invalid MBTI type' in str(exc_info.value)
    
    def test_encode_empty_string(self):
        """Should raise ValueError for empty string."""
        with pytest.raises(ValueError):
            encode_mbti('')
    
    def test_encode_all_valid_types(self):
        """All 16 valid MBTI types should encode successfully."""
        for mbti_type in VALID_MBTI_TYPES:
            result = encode_mbti(mbti_type)
            assert len(result) == 4
            assert all(key in result for key in ['IE', 'NS', 'TF', 'JP'])
            assert all(val in (0, 1) for val in result.values())


class TestDecodeMbti:
    """Tests for decode_mbti function."""
    
    def test_decode_intj(self):
        """All zeros should decode to INTJ."""
        result = decode_mbti({'IE': 0, 'NS': 0, 'TF': 0, 'JP': 0})
        assert result == 'INTJ'
    
    def test_decode_enfp(self):
        """E=1, N=0, F=1, P=1 should decode to ENFP."""
        result = decode_mbti({'IE': 1, 'NS': 0, 'TF': 1, 'JP': 1})
        assert result == 'ENFP'
    
    def test_decode_esfp(self):
        """All ones should decode to ESFP."""
        result = decode_mbti({'IE': 1, 'NS': 1, 'TF': 1, 'JP': 1})
        assert result == 'ESFP'
    
    def test_decode_missing_key(self):
        """Should raise ValueError when key is missing."""
        with pytest.raises(ValueError) as exc_info:
            decode_mbti({'IE': 0, 'NS': 0, 'TF': 0})
        assert 'Missing required keys' in str(exc_info.value)
    
    def test_decode_invalid_value(self):
        """Should raise ValueError for non-binary values."""
        with pytest.raises(ValueError) as exc_info:
            decode_mbti({'IE': 0, 'NS': 0, 'TF': 0, 'JP': 2})
        assert 'Must be 0 or 1' in str(exc_info.value)


class TestRoundTrip:
    """Tests for encode/decode round-trip consistency."""
    
    def test_roundtrip_all_types(self):
        """Encoding then decoding should return the original type."""
        for mbti_type in VALID_MBTI_TYPES:
            encoded = encode_mbti(mbti_type)
            decoded = decode_mbti(encoded)
            assert decoded == mbti_type


class TestListFunctions:
    """Tests for list-based encode/decode functions."""
    
    def test_encode_list_intj(self):
        """INTJ should encode to [0, 0, 0, 0]."""
        result = encode_mbti_list('INTJ')
        assert result == [0, 0, 0, 0]
    
    def test_encode_list_esfp(self):
        """ESFP should encode to [1, 1, 1, 1]."""
        result = encode_mbti_list('ESFP')
        assert result == [1, 1, 1, 1]
    
    def test_decode_list_intj(self):
        """[0, 0, 0, 0] should decode to INTJ."""
        result = decode_mbti_list([0, 0, 0, 0])
        assert result == 'INTJ'
    
    def test_decode_list_esfp(self):
        """[1, 1, 1, 1] should decode to ESFP."""
        result = decode_mbti_list([1, 1, 1, 1])
        assert result == 'ESFP'
    
    def test_decode_list_wrong_length(self):
        """Should raise ValueError for wrong list length."""
        with pytest.raises(ValueError) as exc_info:
            decode_mbti_list([0, 0, 0])
        assert 'Expected 4 values' in str(exc_info.value)
    
    def test_list_roundtrip_all_types(self):
        """List encode then decode should return the original type."""
        for mbti_type in VALID_MBTI_TYPES:
            encoded = encode_mbti_list(mbti_type)
            decoded = decode_mbti_list(encoded)
            assert decoded == mbti_type
