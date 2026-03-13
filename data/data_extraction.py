import pandas as pd
import os
import sys

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

df1 = pd.read_csv('dataset_kaggle_liar/train.tsv', sep='\t')
df1_bis = pd.read_csv('dataset_kaggle_liar/valid.tsv', sep='\t')
df2 = pd.read_csv('fake_news_detection_tweeter/train.csv')
df3 = pd.read_csv('fake_news_detection_UoVictoria/Fake.csv')
df4 = pd.read_csv('fake_news_detection_UoVictoria/True.csv')

df1 = pd.concat([df1, df1_bis], axis=0)

df1_subset = df1[['statement', 'label']].copy()
df1_subset.rename(columns={'statement': 'text'}, inplace=True)
df1_subset = df1_subset[df1_subset['label'] != 'half-true']

dictionnaire_labels = {
    'pants-fire': 0,  
    'false': 0,       
    'barely-true': 0, 
    'mostly-true': 1, 
    'true': 1         
}

df1_subset['label'] = df1_subset['label'].map(dictionnaire_labels)

df1_subset.dropna(subset=['label'], inplace=True)
df1_subset['label'] = df1_subset['label'].astype(int)

df2_subset = df2[['text', 'target']].copy()
df2_subset.rename(columns={'target': 'label'}, inplace=True)

df3['label'] = 1
df4['label'] = 0

df34 = pd.concat([df3, df4], axis=0)
df34 = df34.sample(frac=1, random_state=42).reset_index(drop=True)

df34['raw_text'] = df34['title'] + " " + df34['text']

df34 = df34[['raw_text', 'label']]
df34 = df34.rename(columns={'raw_text': 'text'})

df34['text'] = df34['text'].replace(r'\n|\r|\t', ' ', regex=True)

df_final = pd.concat([df1_subset, df2_subset, df34], axis=0)

df_final.dropna(subset=['text', 'label'], inplace=True)

df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

df_final.to_csv('dataset.csv', index=False)

print(f"\nDONE! Fusion of dataset have been saved to: dataset.csv\n")
