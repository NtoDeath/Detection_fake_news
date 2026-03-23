"""Stylometric feature extraction test utility.

Demonstrates feature extraction pipeline on arbitrary text.
Shows intermediate steps: text normalization and computed metrics.

Usage
-----
python print_features.py
Then edit text_test variable to analyze different inputs.

Dependencies
------------
- style_extractor: StyleExtractor class for feature computation
"""

from typing import Optional, Dict, Any
from style_extractor import StyleExtractor


def print_results(raw_text: Optional[str]) -> None:
    """Extract and display stylometric features for input text.
    
    Pipeline: Raw text → Normalize (entity masking) → Extract 20+ metrics.
    Prints intermediate outputs at each step.
    
    Parameters
    ----------
    raw_text : str or None
        Input text to analyze.
    
    Returns
    -------
    None
        Prints formatted feature table to stdout.
    
    Notes
    -----
    Validates input before processing. Returns early if empty/invalid.
    Float metrics rounded to 3 decimal places for readability.
    """
    print("NEWS ANALYSIS:\n")
    print(f"Raw text:\n{raw_text}\n")
    
    # Initialize bilingual feature extractor
    extractor: StyleExtractor = StyleExtractor()
    
    # Validate input
    if not raw_text or not isinstance(raw_text, str):
        print("Error: The provided text is empty or invalid.")
        return

    # Text preprocessing: mask URLs, mentions, numbers
    clean_text: str = extractor._normalize_text(raw_text)
    print(f"Normalized text:\n{clean_text}\n")
    
    # Compute 20+ stylometric features
    style_vector: Dict[str, Any] = extractor._extract_metrics(clean_text)
    
    # Display features in formatted table
    print("Style vector (FEATURES):\n")
    
    for key, value in style_vector.items():
        if isinstance(value, float):
            # Float metrics: round to 3 decimals
            print(f"{key:<30} : {round(value, 3)}")
        else:
            # Non-float metrics: display as-is
            print(f"{key:<30} : {value}")
    
    print("\n")


# Test with example text containing social media indicators
text_test: str = "ALERT! You must absolutely read this: https://www.google.com. The government is hiding 10,000 terrible and monstrous things! Wake up immediately @user! #news 😡"

print_results(text_test)