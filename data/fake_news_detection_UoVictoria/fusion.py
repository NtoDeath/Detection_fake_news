import pandas as pd

print("Chargement des fichiers Fake.csv et True.csv...")
df_fake = pd.read_csv("Fake.csv")
df_true = pd.read_csv("True.csv")

# ajout des labels
df_fake['label'] = 1
df_true['label'] = 0

df_complet = pd.concat([df_fake, df_true], axis=0)
df_complet = df_complet.sample(frac=1, random_state=42).reset_index(drop=True)

# fusion Titre + Texte
df_complet['texte_brut'] = df_complet['title'] + " " + df_complet['text']

df_final = df_complet[['texte_brut', 'label']]

df_final = df_final.rename(columns={'texte_brut': 'text'})
df_final['text'] = df_final['text'].replace(r'\n|\r|\t', ' ', regex=True)

fichier_sortie = "dataset_pret.csv"
df_final.to_csv(fichier_sortie, index=False, sep='\t')
print(f"\nTERMINÉ ! Le dataset est prêt et sauvegardé sous : {fichier_sortie}")