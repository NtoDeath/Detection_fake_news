"""Feature extraction pipeline for fake news detection.

Extracts stylometric features from corpus using StyleExtractor.
Merges features with original dataset and exports augmented CSV.

Dependencies
------------
- pandas: data manipulation
- style_extractor: bilingual feature computation (20+ metrics)
- tqdm: progress visualization
"""

import pandas as pd
import os
import sys
from typing import Optional
from tqdm import tqdm

from style_extractor import StyleExtractor

input_file: str = "../data/splits/dataset_partA.csv"

# Validate input file existence before processing
if not os.path.exists(input_file):
    print(f"CRITICAL ERROR: Cannot find {input_file}. Are you running the script from the right directory?")
    sys.exit(1)

print(f"Loading data from {input_file}...")
df: pd.DataFrame = pd.read_csv(input_file)

# Preprocessing: handle missing values in text column
# Empty strings prevent feature extraction from crashing on NaN
text_column: str = 'text'
df[text_column] = df[text_column].fillna("")

tqdm.pandas(desc="Extracting Style Features")

# Initialize bilingual feature extractor (loads spaCy EN/FR pipelines)
print("\nInitializing NLP engine (spaCy + TextBlob/VADER)...")
extractor: 'StyleExtractor' = StyleExtractor()

# Compute 20+ stylometric features per text
df_metrics: pd.DataFrame = extractor.transform(df[text_column])

# Merge computed features with original dataset
df_final: pd.DataFrame = pd.concat([df, df_metrics], axis=1)

output_file: str = "../data/complete_train.csv"

# Create output directory if it does not exist
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Export augmented dataset (text + label + 20+ computed features)
df_final.to_csv(output_file, index=False, encoding='utf-8')

print(f"\nDONE! Features have been saved to: {output_file}")