import os
import sys
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from scipy.special import softmax
import evaluate
import torch

if torch.backends.mps.is_available():
    print("Apple Silicon (MPS) detected! Training will be hardware-accelerated.")
    device = torch.device("mps")
elif torch.cuda.is_available():
    print("Nvidia GPU detected! Training will be hardware-accelerated.")
    device = torch.device("cuda")
else:
    print("No hardware acceleration detected, using CPU. Training will be slow.")
    device = torch.device("cpu")

print("Loading CSV files...")
df_A = pd.read_csv("../data/block_A_roberta_train.csv")
df_B = pd.read_csv("../data/block_B_train.csv")
df_C = pd.read_csv("../data/block_C_final_test.csv")

dataset_A = Dataset.from_pandas(df_A[['text', 'label']])
dataset_B = Dataset.from_pandas(df_B[['text', 'label']])
dataset_C = Dataset.from_pandas(df_C[['text', 'label']])

model_name = "distilroberta-base"
print(f"Downloading Tokenizer for {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_text(examples):
    # Truncates overly long texts to 512 tokens (RoBERTa's limit)
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

print("Word tokenization in progress... This may take a minute.")
tokenized_A = dataset_A.map(tokenize_text, batched=True)
tokenized_B = dataset_B.map(tokenize_text, batched=True)
tokenized_C = dataset_C.map(tokenize_text, batched=True)

print("Downloading the Brain (Pre-trained Model)...")
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.to(device)

metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
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
    training_args.fp16 = True

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_A,
    eval_dataset=tokenized_B,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)] # Modify if it stops too early
)

print("\nLaunching training...\n")
trainer.train()

print("\nTraining completed. Saving the model...")
trainer.save_model("./roberta_fine_tunned")

print("\nGenerating probabilities (Super-Feature) for XGBoost...")

predictions_B = trainer.predict(tokenized_B)
predictions_C = trainer.predict(tokenized_C)

fake_proba_B = softmax(predictions_B.predictions, axis=1)[:, 1]
fake_proba_C = softmax(predictions_C.predictions, axis=1)[:, 1]

df_B['roberta_proba'] = fake_proba_B
df_C['roberta_proba'] = fake_proba_C

df_B.to_csv("../data/block_B_train_WITH_PROB.csv", index=False)
os.remove("../data/block_B_train.csv")
df_C.to_csv("../data/block_C_final_test_WITH_PROB.csv", index=False)
os.remove("../data/block_C_final_test.csv")

print("\nBlocks B and C have been enriched with RoBERTa's opinion.")

if sys.platform == "darwin": # macOS
    os.system("afplay /System/Library/Sounds/Glass.aiff")
elif sys.platform == "win32": # Windows
    import winsound
    winsound.MessageBeep(winsound.MB_ICONASTERISK)
else: # Linux
    print('\a') # Terminal bell