import pandas as pd
from sklearn.model_selection import train_test_split

print("Chargement du dataset complet (Style + Texte)...")
df_complet = pd.read_csv("../data/train_complet.csv")

df_roberta_train, df_reste = train_test_split(
    df_complet, 
    test_size=0.40, 
    random_state=42, 
    stratify=df_complet['label'] #  on veut autant de vrais que de faux partout
)

df_xgb_train, df_test_final = train_test_split(
    df_reste, 
    test_size=0.50, 
    random_state=42, 
    stratify=df_reste['label']
)

print("\nRépartition des données :")
print(f"Bloc A (RoBERTa Train) : {len(df_roberta_train)} lignes")
print(f"Bloc B (XGBoost Train) : {len(df_xgb_train)} lignes")
print(f"Bloc C (Test Final)    : {len(df_test_final)} lignes")

df_roberta_train.to_csv("../data/bloc_A_roberta_train.csv", index=False)
df_xgb_train.to_csv("../data/bloc_B_xgb_train.csv", index=False)
df_test_final.to_csv("../data/bloc_C_test_final.csv", index=False)

print("\nDécoupage terminé et fichiers sauvegardés !")