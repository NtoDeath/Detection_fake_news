import pandas as pd
from sklearn.model_selection import train_test_split

print("Loading the complete dataset (Style + Text)...")
df_complete = pd.read_csv("../data/complete_train.csv")

df_roberta_train, df_remainder = train_test_split(
    df_complete, 
    test_size=0.40, 
    random_state=42, 
    stratify=df_complete['label'] # We want as many real as fake news everywhere
)

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
print(f"Block A: \n{df_roberta_train['label'].value_counts(normalize=True).round(3)}")
print(f"Block B: \n{df_xgb_train['label'].value_counts(normalize=True).round(3)}")
print(f"Block C: \n{df_final_test['label'].value_counts(normalize=True).round(3)}")

df_roberta_train.to_csv("../data/block_A_roberta_train.csv", index=False)
df_xgb_train.to_csv("../data/block_B_train.csv", index=False)
df_final_test.to_csv("../data/block_C_final_test.csv", index=False)

print("\nSplitting complete and files saved!")