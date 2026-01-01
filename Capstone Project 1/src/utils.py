"""
MBTI Label Encoding Utilities

This module provides functions for encoding and decoding MBTI personality types
to/from binary representations for machine learning classification.

MBTI Dimensions:
- I/E: Introvert (0) / Extrovert (1)
- N/S: Intuitive (0) / Sensing (1)
- T/F: Thinking (0) / Feeling (1)
- J/P: Judging (0) / Perceiving (1)
"""

from typing import Dict, List, Tuple, Union

# Valid MBTI types (all 16 combinations)
VALID_MBTI_TYPES = [
    'INTJ', 'INTP', 'ENTJ', 'ENTP',
    'INFJ', 'INFP', 'ENFJ', 'ENFP',
    'ISTJ', 'ISTP', 'ESTJ', 'ESTP',
    'ISFJ', 'ISFP', 'ESFJ', 'ESFP'
]

# Dimension mappings: first letter -> 0, second letter -> 1
DIMENSION_MAPPINGS = {
    'IE': {'I': 0, 'E': 1},
    'NS': {'N': 0, 'S': 1},
    'TF': {'T': 0, 'F': 1},
    'JP': {'J': 0, 'P': 1}
}

# Reverse mappings for decoding
REVERSE_MAPPINGS = {
    'IE': {0: 'I', 1: 'E'},
    'NS': {0: 'N', 1: 'S'},
    'TF': {0: 'T', 1: 'F'},
    'JP': {0: 'J', 1: 'P'}
}


def encode_mbti(mbti_type: str) -> Dict[str, int]:
    """
    Convert a 4-letter MBTI type string to 4 binary values.
    
    Args:
        mbti_type: A valid 4-letter MBTI type (e.g., 'INTJ', 'ENFP')
        
    Returns:
        Dictionary with keys 'IE', 'NS', 'TF', 'JP' and binary values (0 or 1)
        
    Raises:
        ValueError: If mbti_type is not a valid 4-letter MBTI type
        
    Example:
        >>> encode_mbti('INTJ')
        {'IE': 0, 'NS': 0, 'TF': 0, 'JP': 0}
        >>> encode_mbti('ENFP')
        {'IE': 1, 'NS': 0, 'TF': 1, 'JP': 1}
    """
    mbti_type = mbti_type.upper().strip()
    
    if mbti_type not in VALID_MBTI_TYPES:
        raise ValueError(
            f"Invalid MBTI type: '{mbti_type}'. "
            f"Must be one of: {', '.join(VALID_MBTI_TYPES)}"
        )
    
    return {
        'IE': DIMENSION_MAPPINGS['IE'][mbti_type[0]],
        'NS': DIMENSION_MAPPINGS['NS'][mbti_type[1]],
        'TF': DIMENSION_MAPPINGS['TF'][mbti_type[2]],
        'JP': DIMENSION_MAPPINGS['JP'][mbti_type[3]]
    }


def decode_mbti(encoded: Dict[str, int]) -> str:
    """
    Convert 4 binary values back to a 4-letter MBTI type string.
    
    Args:
        encoded: Dictionary with keys 'IE', 'NS', 'TF', 'JP' and binary values (0 or 1)
        
    Returns:
        A 4-letter MBTI type string (e.g., 'INTJ', 'ENFP')
        
    Raises:
        ValueError: If encoded dict is missing required keys or has invalid values
        
    Example:
        >>> decode_mbti({'IE': 0, 'NS': 0, 'TF': 0, 'JP': 0})
        'INTJ'
        >>> decode_mbti({'IE': 1, 'NS': 0, 'TF': 1, 'JP': 1})
        'ENFP'
    """
    required_keys = ['IE', 'NS', 'TF', 'JP']
    
    # Validate all required keys are present
    missing_keys = [key for key in required_keys if key not in encoded]
    if missing_keys:
        raise ValueError(f"Missing required keys: {missing_keys}")
    
    # Validate all values are binary (0 or 1)
    for key in required_keys:
        if encoded[key] not in (0, 1):
            raise ValueError(
                f"Invalid value for '{key}': {encoded[key]}. Must be 0 or 1."
            )
    
    return (
        REVERSE_MAPPINGS['IE'][encoded['IE']] +
        REVERSE_MAPPINGS['NS'][encoded['NS']] +
        REVERSE_MAPPINGS['TF'][encoded['TF']] +
        REVERSE_MAPPINGS['JP'][encoded['JP']]
    )


def encode_mbti_list(mbti_type: str) -> List[int]:
    """
    Convert a 4-letter MBTI type string to a list of 4 binary values.
    
    This is a convenience function that returns a list instead of a dict,
    useful for creating numpy arrays or pandas columns.
    
    Args:
        mbti_type: A valid 4-letter MBTI type (e.g., 'INTJ', 'ENFP')
        
    Returns:
        List of 4 binary values in order [IE, NS, TF, JP]
        
    Example:
        >>> encode_mbti_list('INTJ')
        [0, 0, 0, 0]
        >>> encode_mbti_list('ENFP')
        [1, 0, 1, 1]
    """
    encoded = encode_mbti(mbti_type)
    return [encoded['IE'], encoded['NS'], encoded['TF'], encoded['JP']]


def decode_mbti_list(encoded: List[int]) -> str:
    """
    Convert a list of 4 binary values back to a 4-letter MBTI type string.
    
    Args:
        encoded: List of 4 binary values in order [IE, NS, TF, JP]
        
    Returns:
        A 4-letter MBTI type string (e.g., 'INTJ', 'ENFP')
        
    Raises:
        ValueError: If list doesn't have exactly 4 elements or has invalid values
        
    Example:
        >>> decode_mbti_list([0, 0, 0, 0])
        'INTJ'
        >>> decode_mbti_list([1, 0, 1, 1])
        'ENFP'
    """
    if len(encoded) != 4:
        raise ValueError(f"Expected 4 values, got {len(encoded)}")
    
    return decode_mbti({
        'IE': encoded[0],
        'NS': encoded[1],
        'TF': encoded[2],
        'JP': encoded[3]
    })
