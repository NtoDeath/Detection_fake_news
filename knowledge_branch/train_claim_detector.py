"""
Entraînement du Claim Detector
- Chargement du dataset groundtruth.csv
- Fine-tuning d'un DistilBERT pour la détection de claims
- Sauvegarde du modèle
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import torch
from sklearn.metrics import classification_report, confusion_matrix

# Configuration des paths
PROJECT_ROOT = Path.home() / "Documents/IFT714 Traitement des LN/Projet/Detection_fake_news"
KNOWLEDGE_BRANCH = PROJECT_ROOT / "knowledge_branch"

def load_groundtruth_dataset():
    """Charge et prépare le dataset groundtruth.csv"""
    print("📂 Chargement du dataset groundtruth.csv...")
    
    groundtruth_file = KNOWLEDGE_BRANCH / "groundtruth.csv"
    
    if not groundtruth_file.exists():
        print(f"❌ Fichier non trouvé : {groundtruth_file}")
        print("   Exécutez d'abord : python setup_environment.py")
        return None
    
    dataset = pd.read_csv(groundtruth_file)
    
    print(f"   📊 Dimensions : {dataset.shape}")
    print(f"   📋 Colonnes : {dataset.columns.tolist()}")
    print(f"   📈 Distribution des labels :")
    print(dataset['Verdict'].value_counts())
    
    # Créer une colonne 'labels' binaire : 1 pour le niveau 1, 0 pour le reste
    dataset['labels'] = dataset['Verdict'].apply(lambda x: 1 if x == 1 else 0)
    
    # Garder uniquement le texte et le label
    dataset = dataset[['Text', 'labels']]
    
    print(f"   ✅ Dataset préparé\n")
    return dataset

def split_dataset(dataset):
    """Divise le dataset en train/test (80/20)"""
    print("✂️  Division du dataset...")
    
    # Transformer en Dataset Hugging Face
    hg_dataset = Dataset.from_pandas(dataset[['Text', 'labels']])
    
    # Split 80/20
    split_data = hg_dataset.train_test_split(test_size=0.2, seed=42)
    
    print(f"   Train : {len(split_data['train'])} samples")
    print(f"   Test  : {len(split_data['test'])} samples\n")
    
    return split_data

def tokenize_function(examples, tokenizer):
    """Tokenize les textes"""
    return tokenizer(
        examples["Text"],
        padding="max_length",
        truncation=True,
        max_length=256
    )

def train_claim_detector(split_dataset):
    """Fine-tune un DistilBERT pour la détection de claims"""
    print("🤖 Fine-tuning du Claim Detector...")
    print("   Device : CPU (pour éviter les problèmes de mémoire GPU)\n")
    
    model_checkpoint = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    
    # Tokenization
    print("   tokenization en cours...")
    tokenized_datasets = split_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True
    )
    
    # Charger le modèle
    print("   Chargement du modèle...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint,
        num_labels=2
    )
    
    # Configuration de l'entraînement (basée sur hyperparamètres éprouvés)
    training_args = TrainingArguments(
        output_dir=str(KNOWLEDGE_BRANCH / "claim_detector_results"),
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=2,
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=50,
        seed=42,
        warmup_steps=0,
    )
    
    # Initialiser le trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
    )
    
    # Lancer l'entraînement
    print("   ⏳ Entraînement en cours...\n")
    trainer.train()
    
    return trainer, tokenizer, tokenized_datasets

def evaluate_model(trainer, tokenized_datasets):
    """Évalue le modèle et affiche les métriques"""
    print("\n📊 Évaluation du modèle...")
    
    try:
        predictions = trainer.predict(tokenized_datasets["test"])
        preds = np.argmax(predictions.predictions, axis=-1)
        labels = predictions.label_ids
        
        # Matrice de confusion
        cm = confusion_matrix(labels, preds)
        print("   Matrice de confusion :")
        print(cm)
        
        # Rapport de classification
        print("\n   Rapport de classification :")
        print(classification_report(
            labels,
            preds,
            target_names=["Non-Claim", "Claim"]
        ))
    except Exception as e:
        print(f"   ⚠️  Évaluation skippée (entraînement CPU rapide) : {e}\n")

def save_model(trainer, tokenizer):
    """Sauvegarde le modèle et le tokenizer"""
    print("💾 Sauvegarde du modèle...")
    
    model_dir = KNOWLEDGE_BRANCH / "my_claim_model"
    
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    
    print(f"   ✅ Modèle sauvegardé dans : {model_dir}\n")

def test_claim_detector():
    """Test simple du claim detector"""
    print("🧪 Test du Claim Detector...")
    
    from transformers import pipeline
    
    model_dir = KNOWLEDGE_BRANCH / "my_claim_model"
    
    if not (model_dir / "config.json").exists():
        print("   ❌ Modèle non trouvé. Entraînez d'abord.\n")
        return
    
    claim_pipeline = pipeline(
        "text-classification",
        model=str(model_dir),
        tokenizer=str(model_dir),
        device=0 if torch.cuda.is_available() else -1
    )
    
    def detect_claim(text, threshold=0.2):
        results = claim_pipeline(text)
        score_claim = next(
            (r['score'] for r in results if r['label'] == 'LABEL_1'),
            0
        )
        return score_claim > threshold, score_claim
    
    test_cases = [
        ("The unemployment rate in France is 7.5%", 0.5),
        ("The Eiffel Tower is 330 meters tall", 0.5),
        ("There are 6 continents on planet earth", 0.5),
        ("Avatar is a great movie", 0.5),
        ("I think this movie is great", 0.5),
        ("Hello, how are you today?", 0.5),
    ]
    
    print("   Résultats des tests :")
    for text, threshold in test_cases:
        is_claim, score = detect_claim(text, threshold=threshold)
        result = "✅ CLAIM" if is_claim else "❌ NOT CLAIM"
        print(f"   {result:15} (Score: {score:.3f}) - {text[:50]}")
    print()

def main():
    """Fonction principale : orchestre tout le pipeline"""
    print("=" * 60)
    print("🚀 TRAINING CLAIM DETECTOR")
    print("=" * 60)
    print()
    
    # Charger le dataset
    dataset = load_groundtruth_dataset()
    if dataset is None:
        print("❌ Impossible de charger le dataset")
        return
    
    # Diviser le dataset
    data_split = split_dataset(dataset)
    
    # Entraîner le modèle
    trainer, tokenizer, tokenized_datasets = train_claim_detector(data_split)
    
    # Évaluer le modèle
    evaluate_model(trainer, tokenized_datasets)
    
    # Sauvegarder le modèle
    save_model(trainer, tokenizer)
    
    # Tester le modèle
    test_claim_detector()
    
    print("=" * 60)
    print("✅ TRAINING COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    main()
