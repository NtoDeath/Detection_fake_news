import pandas as pd
from tqdm import tqdm

from style_extractor import StyleExtractor

input_file = "../data/fake_news_detection_UoVictoria/dataset_ready.csv"
print(f"Loading data from {input_file}...")

df = pd.read_csv(input_file, sep='\t')

# Prevent the script from crashing if text cells are empty (NaN)
text_column = 'text'
df[text_column] = df[text_column].fillna("")

tqdm.pandas(desc="Extraction du Style en cours")

print("\nInitializing NLP engine (spaCy + TextBlob/VADER)...")
extractor = StyleExtractor()

# Transform all rows into statistical features
df_metrics = extractor.transform(df[text_column])

# Merge the original DataFrame with the extracted features
df_final = pd.concat([df, df_metrics], axis=1)

output_file = "../data/train_complete.csv"
df_final.to_csv(output_file, index=False, encoding='utf-8')

print(f"\nDONE! Features have been saved to: {output_file}")
