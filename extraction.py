import pandas as pd
from tqdm import tqdm

from traitement import traitement_news

fichier_entree = "data/fake_news_detection_UoVictoria/dataset_pret.csv"
print(f"Chargement des données depuis {fichier_entree}...")

df = pd.read_csv(fichier_entree, sep='\t')

# Si certaines cases de la colonne texte sont vides (NaN), 
# on les remplace par une chaîne vide "" pour ne pas faire planter le script.
colonne_texte = 'text'
df[colonne_texte] = df[colonne_texte].fillna("")


def appliquer_pipeline(texte_ligne):
    # parse_news renvoie (texte_normalise, vecteur_style)
    _, dictionnaire_metriques = traitement_news(texte_ligne)
    
    return pd.Series(dictionnaire_metriques)


tqdm.pandas(desc="Extraction du Style en cours")

print("\nLancement du moteur NLP (spaCy + TextBlob/VADER)...")
df_metriques = df[colonne_texte].progress_apply(appliquer_pipeline)

df_final = pd.concat([df, df_metriques], axis=1)

fichier_sortie = "data/train_complet.csv"
df_final.to_csv(fichier_sortie, index=False, encoding='utf-8')

print(f"\nTERMINÉ ! Les features ont été sauvegardées dans : {fichier_sortie}")
