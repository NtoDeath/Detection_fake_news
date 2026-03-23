"""RoBERTa fine-tuning on fake news detection (Block A).

Three-stage training pipeline:
1. Fine-tune distilroberta-base on Block A (60% data, 2 epochs)
2. Generate confidence probabilities on Block B (validation, 20%)
3. Generate probabilities on Block C (test, 20%)

Probabilities added as super-feature for downstream XGBoost ensemble.

Device Detection
----------------
Automatically selects Apple Silicon (MPS) > Nvidia GPU > CPU.

Training Config
---------------
- Model: distilroberta-base (2 labels: fake/true)
- Tokenization: max_length=512, padding/truncation enabled
- Optimizer: AdamW (lr=2e-5)
- Batch size: 8 (train/eval)
- Epochs: 2 with EarlyStopping (patience=5)
- Loss: CrossEntropy (classification)

Dependencies
------------
- torch/transformers: HuggingFace model hub
- datasets: efficient data streaming
- scipy: probability normalization (softmax)
- evaluate: accuracy metric
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from scipy.special import softmax
import evaluate
import torch

# Device detection: prioritize hardware acceleration
if torch.backends.mps.is_available():
    print("Apple Silicon (MPS) detected! Training will be hardware-accelerated.")
    device: torch.device = torch.device("mps")
elif torch.cuda.is_available():
    print("Nvidia GPU detected! Training will be hardware-accelerated.")
    device: torch.device = torch.device("cuda")
else:
    print("No hardware acceleration detected, using CPU. Training will be slow.")
    device: torch.device = torch.device("cpu")

# Load stratified data splits
print("Loading CSV files...")
df_A: pd.DataFrame = pd.read_csv("../data/block_A_roberta_train.csv")
df_B: pd.DataFrame = pd.read_csv("../data/block_B_train.csv")
df_C: pd.DataFrame = pd.read_csv("../data/block_C_final_test.csv")

# Convert pandas DataFrames to HuggingFace Dataset format
dataset_A: Dataset = Dataset.from_pandas(df_A[['text', 'label']])
dataset_B: Dataset = Dataset.from_pandas(df_B[['text', 'label']])
dataset_C: Dataset = Dataset.from_pandas(df_C[['text', 'label']])

model_name: str = "distilroberta-base"
print(f"Downloading Tokenizer for {model_name}...")
tokenizer: 'AutoTokenizer' = AutoTokenizer.from_pretrained(model_name)

def tokenize_text(examples: Dict[str, Any]) -> Dict[str, List[int]]:
    """Tokenize texts with padding and truncation.
    
    Parameters
    ----------
    examples : dict
        Batch of texts with keys 'text', 'label'.
    
    Returns
    -------
    dict
        Tokenized batch with 'input_ids', 'attention_mask'.
        Max length: 512 tokens (RoBERTa standard).
    """
    # Truncate overly long texts to 512 tokens (RoBERTa's limit)
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

print("Word tokenization in progress... This may take a minute.")
tokenized_A: Dataset = dataset_A.map(tokenize_text, batched=True)
tokenized_B: Dataset = dataset_B.map(tokenize_text, batched=True)
tokenized_C: Dataset = dataset_C.map(tokenize_text, batched=True)

print("Downloading the Brain (Pre-trained Model)...")
model: 'AutoModelForSequenceClassification' = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.to(device)

# Load accuracy metric from HuggingFace hub
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred) -> Dict[str, float]:
    """Compute classification accuracy metric.
    
    Parameters
    ----------
    eval_pred : tuple
        (logits, labels) from model predictions.
    
    Returns
    -------
    dict
        Keys: 'accuracy' (float between 0 and 1).
    """
    # Convert raw logits to predicted class labels
    logits, labels = eval_pred
    predictions: np.ndarray = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Configure training hyperparameters for 2-epoch fine-tuning
training_args: TrainingArguments = TrainingArguments(
    output_dir="./roberta_fake_news",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,  # L2 regularization to prevent overfitting
    eval_strategy="epoch",  # Evaluate at end of each epoch
    save_strategy="epoch",  # Save checkpoint after each epoch
    load_best_model_at_end=True,  # Restore best model after training
    save_total_limit=2,  # Keep only 2 most recent checkpoints
    push_to_hub=False,
    logging_steps=100
)

# Enable mixed precision training (FP16) on CUDA devices
if device.type == "cuda":
    training_args.fp16 = True

# Initialize trainer with model, datasets, and callbacks
trainer: Trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_A,
    eval_dataset=tokenized_B,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]  # Stop if no improvement for 5 evals
)

print("\nLaunching training...\n")
trainer.train()

print("\nTraining completed. Saving the model...")
trainer.save_model("./roberta_fine_tunned")

print("\nGenerating probabilities (Super-Feature) for XGBoost...")

# Get raw model predictions (logits) on blocks B and C
predictions_B = trainer.predict(tokenized_B)
predictions_C = trainer.predict(tokenized_C)

# Convert logits to normalized probabilities via softmax
# fake_proba contains probability of being fake news (column 1)
fake_proba_B: np.ndarray = softmax(predictions_B.predictions, axis=1)[:, 1]
fake_proba_C: np.ndarray = softmax(predictions_C.predictions, axis=1)[:, 1]

# Append RoBERTa confidence as super-feature for XGBoost
df_B['roberta_proba'] = fake_proba_B
df_C['roberta_proba'] = fake_proba_C

# Save augmented blocks and remove old versions
df_B.to_csv("../data/block_B_train_WITH_PROB.csv", index=False)
os.remove("../data/block_B_train.csv")
df_C.to_csv("../data/block_C_final_test_WITH_PROB.csv", index=False)
os.remove("../data/block_C_final_test.csv")

print("\nBlocks B and C have been enriched with RoBERTa's opinion.")

# Cross-platform notification sound
if sys.platform == "darwin":  # macOS
    os.system("afplay /System/Library/Sounds/Glass.aiff")
elif sys.platform == "win32":  # Windows
    import winsound
    winsound.MessageBeep(winsound.MB_ICONASTERISK)
else:  # Linux
    print('\a')  # Terminal bell