"""
Text Preprocessor for MBTI Personality Prediction.

This module provides text cleaning and preprocessing functionality
for preparing raw text data for MBTI personality classification.
"""

import re
from typing import List


class TextPreprocessor:
    """
    Preprocessor for cleaning and normalizing text data for MBTI prediction.
    
    Handles:
    - URL removal
    - MBTI type mention removal (to prevent data leakage)
    - Special character removal
    - Whitespace normalization
    - Lowercase conversion
    """
    
    # All 16 valid MBTI types
    MBTI_TYPES = [
        'INTJ', 'INTP', 'ENTJ', 'ENTP',
        'INFJ', 'INFP', 'ENFJ', 'ENFP',
        'ISTJ', 'ISFJ', 'ESTJ', 'ESFJ',
        'ISTP', 'ISFP', 'ESTP', 'ESFP'
    ]
    
    # Regex pattern for URLs
    URL_PATTERN = re.compile(
        r'https?://\S+|www\.\S+',
        re.IGNORECASE
    )
    
    # Regex pattern for MBTI types (case insensitive, word boundaries)
    MBTI_PATTERN = re.compile(
        r'\b(' + '|'.join(MBTI_TYPES) + r')\b',
        re.IGNORECASE
    )
    
    # Pattern for special characters (keep alphanumeric and basic punctuation)
    SPECIAL_CHARS_PATTERN = re.compile(r'[^a-zA-Z0-9\s.,!?\'"-]')
    
    # Pattern for multiple whitespace
    WHITESPACE_PATTERN = re.compile(r'\s+')
    
    def clean_text(self, text: str) -> str:
        """
        Clean raw text by removing URLs, MBTI mentions, special characters,
        and normalizing whitespace.
        
        Args:
            text: Raw input text string
            
        Returns:
            Cleaned and normalized text string (lowercase)
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Remove URLs
        text = self.URL_PATTERN.sub(' ', text)
        
        # Remove MBTI type mentions (to prevent data leakage)
        text = self.MBTI_PATTERN.sub(' ', text)
        
        # Remove special characters
        text = self.SPECIAL_CHARS_PATTERN.sub(' ', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Normalize whitespace (replace multiple spaces with single space)
        text = self.WHITESPACE_PATTERN.sub(' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def preprocess_posts(self, posts: str, delimiter: str = "|||") -> str:
        """
        Process concatenated posts string by splitting, cleaning each post,
        and rejoining.
        
        Args:
            posts: Concatenated posts string separated by delimiter
            delimiter: Separator between posts (default: "|||")
            
        Returns:
            Cleaned and rejoined posts string
        """
        if not posts or not isinstance(posts, str):
            return ""
        
        # Split by delimiter
        post_list = posts.split(delimiter)
        
        # Clean each post
        cleaned_posts = [self.clean_text(post) for post in post_list]
        
        # Filter out empty posts
        cleaned_posts = [post for post in cleaned_posts if post]
        
        # Rejoin with space
        return " ".join(cleaned_posts)
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """
        Preprocess a batch of text strings.
        
        Args:
            texts: List of raw text strings
            
        Returns:
            List of cleaned text strings
        """
        return [self.preprocess_posts(text) for text in texts]
