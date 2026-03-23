"""Stratified data split for three-stage fake news detection pipeline.

Divides augmented dataset into three blocks:
- Block A (60%): RoBERTa fine-tuning
- Block B (20%): XGBoost ensemble training
- Block C (20%): Final evaluation

Stratification ensures balanced label distribution (Fake/True) across blocks.

Dependencies
------------
- pandas: data manipulation
- sklearn: stratified train_test_split
"""

import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split

print("Loading the complete dataset (Style + Text)...")
df_complete: pd.DataFrame = pd.read_csv("../data/complete_train.csv")

# First split: 60% for RoBERTa fine-tuning, 40% remainder
# Stratify ensures proportional Fake/True distribution in Block A
df_roberta_train: pd.DataFrame
df_remainder: pd.DataFrame
df_roberta_train, df_remainder = train_test_split(
    df_complete, 
    test_size=0.40, 
    random_state=42, 
    stratify=df_complete['label']  # Preserve Fake/True ratio
)

# Second split: divide remainder into 20% XGB training, 20% final test
# Stratification maintains label balance for fair model comparison
df_xgb_train: pd.DataFrame
df_final_test: pd.DataFrame
df_xgb_train, df_final_test = train_test_split(
    df_remainder, 
    test_size=0.50, 
    random_state=42, 
    stratify=df_remainder['label']
)

print("\nData distribution:")
print(f"Block A (RoBERTa Train) : {len(df_roberta_train)} rows")
print(f"Block B (Training) : {len(df_xgb_train)} rows")
print(f"Block C (Final Test)    : {len(df_final_test)} rows")

print("\nSanity Check - Class Distribution (Fake=1, True=0):")
print(f"Block A (RoBERTa): \n{df_roberta_train['label'].value_counts(normalize=True).round(3)}")
print(f"Block B (XGBoost): \n{df_xgb_train['label'].value_counts(normalize=True).round(3)}")
print(f"Block C (Test):    \n{df_final_test['label'].value_counts(normalize=True).round(3)}")

# Export stratified splits to CSV files for downstream training
df_roberta_train.to_csv("../data/block_A_roberta_train.csv", index=False)
df_xgb_train.to_csv("../data/block_B_train.csv", index=False)
df_final_test.to_csv("../data/block_C_final_test.csv", index=False)

print("\nSplitting complete and files saved!")