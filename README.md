# Detection\_fake\_news

Repo Git du projet NLP

> **Document** : Analyse technique approfondie de chaque branche, des responsabilités des modules et des interfaces.  
> **Audience** : Développeurs ayant besoin de comprendre le rôle de chaque module Python.  

## Organisation du projet

Le projet contient plusieurs branches :

  - la **knowledge\_branch** contient tous les fichiers .py et .ipynb sur la détection de fake news grâce à la méthode basée sur la connaissance (*knowledge-based*).
  - la **style\_branch** contient tous les fichiers .py et .ipynb sur la détection de fake news grâce à la méthode basée sur le style (*style-based*).
  - la **fusion\_branch** contient tous les fichiers .py et .ipynb pour combiner les deux méthodes.

**Sources de données :**

  - Data news media : [Lien ISOT](https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/)
  - Data Twitter : [Lien Kaggle Twitter](https://www.kaggle.com/competitions/nlp-getting-started/data?select=test.csv)
  - Data LIAR Kaggle : [Lien Kaggle LIAR](https://www.kaggle.com/datasets/doanquanvietnamca/liar-dataset/data)

-----

## Structure par défaut du projet

```
/home/juul/Documents/IFT714 Traitement des LN/Projet/Detection_fake_news/
├─ unified_main.ipynb ⭐               [Orchestrateur]
│
├─ data/
│  ├─ data_extraction.py               [Cœur]
│  ├─ prepare_part_B_heterogeneous.py  [Cœur]
│  ├─ dataset_kaggle_liar/             [Données]
│  ├─ fake_news_detection_tweeter/     [Données]
│  ├─ fake_news_detection_UoVictoria/  [Données]
│  └─ splits/                          [Sorties des Parties A/B]
│
├─ style_branch/
│  ├─ fine_tunning.py                  [Cœur]
│  ├─ model_comp.py                    [Cœur]
│  ├─ roberta_fine_tunned/             [Poids du modèle]
│  └─ results/                         [Résultats]
│
├─ knowledge_branch/
│  ├─ train_claim_detector_partA.py    [Cœur]
│  ├─ full_pipeline.py                 [Cœur]
│  ├─ claim_detector_model/            [Poids du modèle]
│  └─ results/                         [Résultats]
│
├─ fusion_branch/
│  ├─ config.py                        [Point central]
│  ├─ 0[0-8]_*.py                      [Orchestration]
│  ├─ Diagram_fusion.pdf               [Diagramme de la fusion]
│  └─ results/                         [Résultats]
│
└─ cli_tool/
   ├─ main.py                          [Entrée]
   ├─ model_loaders.py                 [Cœur]
   ├─ models/                          [Modèles déployés]
   ├─ setup.py                         [Déploiement]
   └─ Detection_fake_news_CLI_Tool.zip [Archive zip de l'outil CLI (avec les models)]
```

-----

## Table des Matières

1.  [Modules du Pipeline de Données](https://www.google.com/search?q=%23modules-du-pipeline-de-donn%C3%A9es)
2.  [Modules de la Branche Style](https://www.google.com/search?q=%23modules-de-la-branche-style)
3.  [Modules de la Branche Connaissance](https://www.google.com/search?q=%23modules-de-la-branche-connaissance)
4.  [Modules de la Branche Fusion](https://www.google.com/search?q=%23modules-de-la-branche-fusion)
5.  [Modules de l'Outil CLI](https://www.google.com/search?q=%23modules-de-loutil-cli)
6.  [Utilitaires Partagés](https://www.google.com/search?q=%23utilitaires-partag%C3%A9s)
7.  [Configuration et Chemins](https://www.google.com/search?q=%23configuration-et-chemins)

-----

## Modules du Pipeline de Données

### `data/data_extraction.py`

**Objectif** : Fusionner 5 sources de données hétérogènes en un jeu de données unifié avec des étiquettes normalisées.

**Entrée** :

```
data/
├─ dataset_kaggle_liar/         [Jeu de données LIAR de Kaggle]
├─ fake_news_detection_tweeter/ [Fake news Twitter de Stanford]
├─ fake_news_detection_UoVictoria/
├─ knowledge_based/             [Jeu de données FEVER]
└─ groundtruth.csv              [Étiquettes ClaimBuster]
```

**Sortie** :

```
data/unified_data/complete_train.csv
├─ text_id (identifiant unique)
├─ text (allégation ou extrait d'actualité)
└─ label (normalisé : 0=FAKE, 1=TRUE)
```

**Fonctions Clés** :

  - `load_liar_data()` : Charge LIAR depuis Kaggle.
  - `load_twitter_data()` : Charge les fake news de Twitter.
  - `load_uovictoria_data()` : Charge le jeu de données UoVictoria.
  - `load_fever_data()` : Charge FEVER (format JSON).
  - `normalize_labels()` : Harmonise les valeurs des étiquettes.
  - `merge_sources()` : Combine le tout en un seul fichier CSV.

**Normalisation des Étiquettes** :

```
Étiquettes d'entrée (par source) :
├─ LIAR : mostly-true, half-true, mostly-false, false
├─ Twitter : REAL (1), FAKE (0)
├─ UoVictoria : 0 (fake), 1 (real)
├─ FEVER : SUPPORTS (1), REFUTES (0), NOT_ENOUGH_INFO (neutre)
└─ ClaimBuster : 0 (not-claim), 1 (claim)

Sortie (unifiée) :
├─ 0 = FAKE / FAUX / RÉFUTÉ
└─ 1 = VRAI / RÉEL / SUPPORTÉ
```

-----

### `data/prepare_part_B_heterogeneous.py`

**Objectif** : Créer des partitions stratifiées uniques et sans chevauchement pour la Partie A (80%) et la Partie B (20%).

**Entrée** :

  - `data/unified_data/complete_train.csv` (données fusionnées)

**Sortie** :

```
data/splits/
├─ dataset_partA.csv           (25.6k échantillons - LIAR+Twitter+UoVictoria)
├─ groundtruth_partA.csv       (ClaimBuster Partie A)
├─ train_partA.jsonl           (Format FEVER Partie A)
├─ part_B_validation.csv       (31.8k échantillons - TOUTES sources combinées)
└─ part_B_metadata.json        (statistiques & documentation)
```

**Pourquoi c'est important** :
Cela **évite la fuite de données (data leakage)** :

  - La Partie A est utilisée exclusivement pour les Phases 1 à 5 (entraînement supervisé).
  - La Partie B est utilisée exclusivement pour la Phase 6 (validation de la fusion).
  - → Évaluation équitable : les modèles ne voient jamais la Partie B pendant l'entraînement.

-----

## Modules de la Branche Style

### `style_branch/style_extractor.py`

**Objectif** : Extraire plus de 20 caractéristiques stylométriques du texte pour l'apprentissage automatique.

**Caractéristiques extraites** :

  - **Lexicales** : taille du vocabulaire, ratio type-token (TTR), longueur moyenne des mots.
  - **Syntaxiques** : longueur moyenne des phrases, ponctuation, ratio de majuscules.
  - **Complexité** : score de lisibilité Flesch-Kincaid, indice Gunning-Fog.
  - **Statistiques** : entropie (densité d'information), distribution de fréquence des mots.

-----

### `style_branch/feature_extraction_partA.py`

**Objectif** : Appliquer `style_extractor.py` à toutes les données de la Partie A et mettre les résultats en cache (fichiers `.pkl`).

-----

### `style_branch/fine_tunning.py`

**Objectif** : Ajuster finement (fine-tune) RoBERTa sur la Partie A pour la classification de fake news.

**Détails de l'entraînement** :

  - Utilise `distilroberta-base` ou `roberta-base`.
  - Boucle d'entraînement de 5 époques avec un taux d'apprentissage de 2e-5.
  - Nécessite un GPU (CUDA recommandé).

-----

### `style_branch/model_comp.py`

**Objectif** : Comparer des ensembles RandomForest vs XGBoost sur les caractéristiques issues de RoBERTa et de la stylométrie.

  - Effectue une recherche par grille (Grid Search) sur 125 configurations pour trouver le meilleur modèle.

-----

### `style_branch/inference_pipeline.py`

**Objectif** : Encapsuleur (wrapper) combinant RoBERTa + le modèle d'ensemble pour l'inférence.

  - Prend un texte brut en entrée et renvoie une prédiction (0 ou 1) avec un score de confiance.

-----

## Modules de la Branche Connaissance (Knowledge)

### `knowledge_branch/claim_detection.py`

**Objectif** : Extraire les allégations (*claims*) du texte en utilisant DistilBERT.

  - Distingue les faits vérifiables des opinions ou questions.

-----

### `knowledge_branch/evidence_retrieval.py`

**Objectif** : Récupérer des preuves (soutien ou réfutation) via les API Google Custom Search et Wolfram Alpha.

-----

### `knowledge_branch/claim_verification.py`

**Objectif** : Vérifier les allégations en utilisant l'inférence en langage naturel (NLI) avec un modèle RoBERTa-large entraîné sur FEVER.

-----

### `knowledge_branch/full_pipeline.py`

**Objectif** : Orchestration de bout en bout : Détection d'allégation → Récupération de preuves → Vérification.

-----

## Modules de la Branche Fusion

### `fusion_branch/config.py`

**Objectif** : Hub central de configuration pour tous les chemins et constantes de la phase de fusion.

-----

### `fusion_branch/0[0-8]_*.py` (Scripts d'orchestration)

Ces scripts s'exécutent séquentiellement pour la Phase 6 :

  - `00_verify_models.py` : Vérifie l'existence de tous les modèles.
  - `01_load_predictions.py` : Génère les prédictions sur la Partie B.
  - `02_split_data.py` : Divise la Partie B en deux (50/50).
  - `03 à 07` : Testent différentes stratégies de fusion (cascade, pondération par confiance, empilement/stacking RF).
  - `08_comparison_visualize.py` : Compare et résume les résultats.

-----

## Modules de l'Outil CLI

### `cli_tool/main.py`

**Objectif** : Interface interactive (REPL) pour l'inférence en production.

  - Permet de tester une phrase en tapant `predict "votre texte ici"`.

-----

### `cli_tool/setup.py`

**Objectif** : Automatisation du déploiement en copiant les modèles entraînés des branches *Style* et *Knowledge* vers le dossier `cli_tool/models/`.

-----

## Utilitaires Partagés

### `unzip.py`

**Objectif** : Décompresser les sources de données téléchargées (LIAR, Twitter, UoVictoria).

-----

## Configuration et Chemins

### Variables d'environnement

Requises pour la branche connaissance :

```bash
export GOOGLE_API_KEY="votre-cle-ici"
export WOLFRAM_APP_ID="votre-id-ici"
```

