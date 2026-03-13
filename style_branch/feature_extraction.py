import pandas as pd
import os
import sys
from tqdm import tqdm

from style_extractor import StyleExtractor

input_file = "../data/dataset.csv"

if not os.path.exists(input_file):
    print(f"CRITICAL ERROR: Cannot find {input_file}. Are you running the script from the right directory?")
    sys.exit(1)

print(f"Loading data from {input_file}...")
df = pd.read_csv(input_file)

# Prevent the script from crashing if text cells are empty (NaN)
text_column = 'text'
df[text_column] = df[text_column].fillna("")

tqdm.pandas(desc="Extracting Style Features")

print("\nInitializing NLP engine (spaCy + TextBlob/VADER)...")
extractor = StyleExtractor()

df_metrics = extractor.transform(df[text_column])

df_final = pd.concat([df, df_metrics], axis=1)

output_file = "../data/complete_train.csv"

# Ensure output directory exists
os.makedirs(os.path.dirname(output_file), exist_ok=True)

df_final.to_csv(output_file, index=False, encoding='utf-8')

print(f"\nDONE! Features have been saved to: {output_file}")