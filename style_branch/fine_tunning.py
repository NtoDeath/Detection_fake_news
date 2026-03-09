import os
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from scipy.special import softmax
import evaluate
import torch

if torch.backends.mps.is_available():
    print("Moteur graphique Apple (MPS) détecté ! L'entraînement sera accéléré.")
    device = torch.device("mps")
elif torch.cuda.is_available():
    print("GPU Nvidia détecté ! L'entraînement sera accéléré.")
    device = torch.device("cuda")
else:
    print("MPS non détecté, utilisation du CPU. L'entraînement sera lent.")
    device = torch.device("cpu")

print("Chargement des fichiers CSV...")
df_A = pd.read_csv("../data/bloc_A_roberta_train.csv")
df_B = pd.read_csv("../data/bloc_B_xgb_train.csv")
df_C = pd.read_csv("../data/bloc_C_test_final.csv")

dataset_A = Dataset.from_pandas(df_A[['text', 'label']])
dataset_B = Dataset.from_pandas(df_B[['text', 'label']])
dataset_C = Dataset.from_pandas(df_C[['text', 'label']])

nom_modele = "distilroberta-base"
print(f"Téléchargement du Tokenizer {nom_modele}...")
tokenizer = AutoTokenizer.from_pretrained(nom_modele)

def tokeniser_texte(exemples):
    # Tronque les textes trop longs à 512 tokens (limite de RoBERTa)
    return tokenizer(exemples["text"], padding="max_length", truncation=True, max_length=512)

print("Découpage des mots (Tokenization) en cours... Cela peut prendre une minute.")
tokenized_A = dataset_A.map(tokeniser_texte, batched=True)
tokenized_B = dataset_B.map(tokeniser_texte, batched=True)
tokenized_C = dataset_C.map(tokeniser_texte, batched=True)

print("Téléchargement du Cerveau (Modèle pré-entraîné)...")
modele = AutoModelForSequenceClassification.from_pretrained(nom_modele, num_labels=2)
modele.to(device)

metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

arguments_entrainement = TrainingArguments(
    output_dir="./roberta_fake_news",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    save_total_limit=2,
    push_to_hub=False,
    logging_steps=100
)
if device.type == "cuda":
    arguments_entrainement.fp16 = True

trainer = Trainer(
    model=modele,
    args=arguments_entrainement,
    train_dataset=tokenized_A,
    eval_dataset=tokenized_B,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)] # à modifier si arret prématuré
)

print("\nLancement de l'entraînement...\n")
trainer.train()

print("\nEntraînement terminé. Sauvegarde du modèle...")
trainer.save_model("./roberta_fine_tunned")

print("\nGénération des probabilités (Super-Feature) pour XGBoost...")

predictions_B = trainer.predict(tokenized_B)
predictions_C = trainer.predict(tokenized_C)

proba_fake_B = softmax(predictions_B.predictions, axis=1)[:, 1]
proba_fake_C = softmax(predictions_C.predictions, axis=1)[:, 1]

df_B['roberta_proba'] = proba_fake_B
df_C['roberta_proba'] = proba_fake_C

df_B.to_csv("../data/bloc_B_xgb_train_AVEC_PROBA.csv", index=False)
df_C.to_csv("../data/bloc_C_test_final_AVEC_PROBA.csv", index=False)

print("\nLes blocs B et C ont été enrichis avec l'avis de RoBERTa.")
os.system("afplay /System/Library/Sounds/Glass.aiff") # it is too long so the sound will make me up when it's done (probably sleeping)
