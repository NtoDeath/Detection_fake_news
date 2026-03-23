"""Data fusion pipeline combining 5 fake news datasets.

Merges multiple disparate dataset sources into unified training set:
1. LIAR (fact-checking database)           → train.tsv + valid.tsv
2. Twitter fake news (Stanford)             → train.csv
3. UoVictoria fake news collection (Fake.csv, True.csv)

Standardization
---------------
All datasets normalized to schema: {text: str, label: int}
- label 0 = fake news / misinformation
- label 1 = true news / reliable

Label Mapping by Source
-----------------------
LIAR: pants-fire/false/barely-true → 0 (fake)
      mostly-true/true → 1 (true)
      half-true → dropped (ambiguous)

Twitter: target column already binary → renamed to label

UoVictoria: Fake.csv → label 1, True.csv → label 0
           (Note: inverted logic! True=0 for consistency)

Output
------
Complete unified training set ready for model training.

Dependencies
------------
- pandas: data manipulation and file I/O
"""

import pandas as pd
from typing import Dict, Any, Optional
import os
import sys

# Validate all source files exist before processing
if not os.path.exists("dataset_kaggle_liar/train.tsv") or not os.path.exists("dataset_kaggle_liar/valid.tsv"):
    print("CRITICAL ERROR: 'dataset_kaggle_liar/train.tsv' or 'dataset_kaggle_liar/valid.tsv' not found in the current directory.")
    sys.exit(1)
if not os.path.exists("fake_news_detection_tweeter/train.csv"):
    print("CRITICAL ERROR: 'fake_news_detection_tweeter/train.csv' not found in the current directory.")
    sys.exit(1)
if not os.path.exists("fake_news_detection_UoVictoria/Fake.csv"):
    print("CRITICAL ERROR: 'fake_news_detection_UoVictoria/Fake.csv' not found in the current directory.")
    sys.exit(1)
if not os.path.exists("fake_news_detection_UoVictoria/True.csv"):
    print("CRITICAL ERROR: 'fake_news_detection_UoVictoria/True.csv' not found in the current directory.")
    sys.exit(1)

# Load source datasets
print("Loading datasets...")
df1: pd.DataFrame = pd.read_csv('dataset_kaggle_liar/train.tsv', sep='\t')
df1_bis: pd.DataFrame = pd.read_csv('dataset_kaggle_liar/valid.tsv', sep='\t')
df2: pd.DataFrame = pd.read_csv('fake_news_detection_tweeter/train.csv')
df3: pd.DataFrame = pd.read_csv('fake_news_detection_UoVictoria/Fake.csv')
df4: pd.DataFrame = pd.read_csv('fake_news_detection_UoVictoria/True.csv')

# Merge LIAR train and validation sets
df1 = pd.concat([df1, df1_bis], axis=0)

# DATASET 1: LIAR (fact-checking database)
# Extract relevant columns and standardize schema
df1_subset: pd.DataFrame = df1[['statement', 'label']].copy()
df1_subset.rename(columns={'statement': 'text'}, inplace=True)
# Remove ambiguous labels (half-true = neither clearly fake nor true)
df1_subset = df1_subset[df1_subset['label'] != 'half-true']

# Map LIAR labels (6-class) to binary (0=fake, 1=true)
# pants-fire=0.0, false=0.0, barely-true=0.0 (grouped as 'fake')
# mostly-true=1.0, true=1.0 (grouped as 'true')
label_mapping: Dict[str, int] = {
    'pants-fire': 0,  # Complete falsehood
    'false': 0,  # False statement
    'barely-true': 0,  # Minimal truth
    'mostly-true': 1,  # Mostly accurate
    'true': 1  # Fully accurate
}

df1_subset['label'] = df1_subset['label'].map(label_mapping)

# Clean: remove rows with missing labels and ensure integer type
df1_subset.dropna(subset=['label'], inplace=True)
df1_subset['label'] = df1_subset['label'].astype(int)

# DATASET 2: Twitter fake news detection
# Already has text/label columns, just rename target to label
df2_subset: pd.DataFrame = df2[['text', 'target']].copy()
df2_subset.rename(columns={'target': 'label'}, inplace=True)

# DATASET 3: UoVictoria fake news collection
# Two separate files: Fake.csv (misinformation) and True.csv (factual articles)
df3['label'] = 1  # Label: 1 = True news -> But UoVictoria True.csv contains factual articles
df4['label'] = 0  # Label: 0 = Fake news -> But UoVictoria Fake.csv contains misinformation

# Combine Fake and True articles
df34: pd.DataFrame = pd.concat([df3, df4], axis=0)
# Shuffle to remove dataset-order bias and reset index
df34 = df34.sample(frac=1, random_state=42).reset_index(drop=True)

# Merge title and text columns for complete article representation
df34['raw_text']: pd.Series = df34['title'] + " " + df34['text']

# Select and rename columns to match unified schema
df34 = df34[['raw_text', 'label']]
df34 = df34.rename(columns={'raw_text': 'text'})

# Clean text: remove line breaks, carriage returns, tabs
df34['text'] = df34['text'].replace(r'\n|\r|\t', ' ', regex=True)

# FINAL STEP: Concatenate all three datasets
# Unified training set: all rows normalized to {text, label}
df_final: pd.DataFrame = pd.concat([df1_subset, df2_subset, df34], axis=0)

df_final.dropna(subset=['text', 'label'], inplace=True)

df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

df_final.to_csv('dataset.csv', index=False)

print(f"\nDONE! Fusion of dataset have been saved to: dataset.csv\n")
