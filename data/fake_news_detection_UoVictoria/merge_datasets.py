import pandas as pd
import os
import sys

if not os.path.exists("Fake.csv") or not os.path.exists("True.csv"):
    print("CRITICAL ERROR: 'Fake.csv' or 'True.csv' not found in the current directory.")
    sys.exit(1)

print("Loading Fake.csv and True.csv files...")
df_fake = pd.read_csv("Fake.csv")
df_true = pd.read_csv("True.csv")

# Add labels
df_fake['label'] = 1
df_true['label'] = 0

df_complete = pd.concat([df_fake, df_true], axis=0)
df_complete = df_complete.sample(frac=1, random_state=42).reset_index(drop=True)

# Merge Title + Text
df_complete['raw_text'] = df_complete['title'] + " " + df_complete['text']

df_final = df_complete[['raw_text', 'label']]
df_final = df_final.rename(columns={'raw_text': 'text'})

# Clean hidden tabs and newlines inside the text
df_final['text'] = df_final['text'].replace(r'\n|\r|\t', ' ', regex=True)

print(f"Dataset merged successfully! Total rows: {len(df_final)}")
print(f"Fake news: {len(df_fake)} | True news: {len(df_true)}")

output_file = "dataset_ready.csv"
df_final.to_csv(output_file, index=False, sep='\t')
print(f"\nDONE! The dataset is ready and saved as: {output_file}")