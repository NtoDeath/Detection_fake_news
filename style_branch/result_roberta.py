"""RoBERTa baseline evaluation on test set (Block C).

Evaluates fine-tuned RoBERTa model separately to compare with ensemble methods.
Uses batched inference for efficiency. Computes comprehensive metrics:
- Accuracy, F1, ROC-AUC, log loss
- Per-class precision/recall/F1

Purpose
-------
Establish baseline performance for fake news detection.
Comparison with ensemble models (Random Forest + XGBoost) validates
ensembling strategy (combining style + semantic features).

Dependencies
------------
- torch: GPU/CPU inference acceleration
- transformers: RoBERTa model and tokenizer
- sklearn.metrics: evaluation functions
- scipy: probability normalization
"""

import torch
import pandas as pd
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss, classification_report
from scipy.special import softmax
from tqdm import tqdm  # Progress bar for batch iteration


def evaluate_roberta_baseline(
    model_path: str = "./roberta_fine_tunned", 
    test_data_path: str = "../data/block_C_final_test_WITH_PROB.csv"
) -> None:
    """Evaluate fine-tuned RoBERTa model on test set.
    
    Performs batched inference on test texts. Computes and displays
    accuracy, F1, ROC-AUC, log loss, and per-class metrics.
    
    Parameters
    ----------
    model_path : str
        Directory containing fine-tuned model checkpoint and tokenizer.
    test_data_path : str
        Path to test CSV with 'text' and 'label' columns.
    
    Returns
    -------
    None
        Prints evaluation results and classification report.
    
    Notes
    -----
    Uses batch size 16 for memory efficiency on GPU/CPU.
    Text truncated to 512 tokens (RoBERTa standard).
    Device detection: MPS (Apple Silicon) > CUDA (Nvidia) > CPU.
    """
    # Automatic device selection: prioritize hardware acceleration
    device: torch.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    # Load pre-trained RoBERTa and tokenizer from checkpoint
    tokenizer: 'AutoTokenizer' = AutoTokenizer.from_pretrained(model_path)
    model: 'AutoModelForSequenceClassification' = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()  # Disable dropout, batch norm tracking for inference

    # Load test data
    df_test: pd.DataFrame = pd.read_csv(test_data_path)
    textes: List[str] = df_test['text'].tolist()
    labels_vrais: List[int] = df_test['label'].tolist()
    
    # Storage for predictions and probabilities
    predictions: List[int] = []
    probabilites_classe_1: List[float] = []

    # Batch processing for efficiency
    batch_size: int = 16 
    
    # Forward pass without gradient computation (inference mode)
    with torch.no_grad():
        for i in tqdm(range(0, len(textes), batch_size), desc="Évaluation RoBERTa"):
            # Extract batch of texts
            batch_textes: List[str] = textes[i:i + batch_size]
            
            # Tokenize with padding/truncation
            inputs: Dict = tokenizer(batch_textes, return_tensors="pt", padding=True, truncation=True, max_length=512)
            # Move tensors to device (GPU/CPU)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Forward pass: get logits
            outputs = model(**inputs)
            logits: 'np.ndarray' = outputs.logits.detach().cpu().numpy()
            
            # Convert logits to probabilities and predictions
            probs: 'np.ndarray' = softmax(logits, axis=1)
            preds: 'np.ndarray' = logits.argmax(axis=-1)
            
            # Accumulate predictions (column 1 = fake news probability)
            predictions.extend(preds)
            probabilites_classe_1.extend(probs[:, 1])

    # Compute evaluation metrics on full test set
    acc: float = accuracy_score(labels_vrais, predictions)
    f1: float = f1_score(labels_vrais, predictions)
    roc_auc: float = roc_auc_score(labels_vrais, probabilites_classe_1)
    ll: float = log_loss(labels_vrais, probabilites_classe_1)

    # Display baseline results
    print("\n" + "="*50)
    print("RÉSULTATS DE ROBERTA SEUL (BASELINE)")
    print("="*50)
    print(f"Accuracy  : {acc * 100:.2f}%")
    print(f"F1 Score  : {f1 * 100:.2f}%")
    print(f"ROC-AUC   : {roc_auc * 100:.2f}%")
    print(f"Log Loss  : {ll * 100:.2f}%")
    print("-" * 50)
    print("\nRapport détaillé :\n")
    # Per-class metrics (precision, recall, F1)
    print(classification_report(labels_vrais, predictions))


if __name__ == "__main__":
    # Run baseline evaluation on Block C (final test set)
    evaluate_roberta_baseline(
        model_path="./roberta_fine_tunned", 
        test_data_path="../data/block_C_final_test_WITH_PROB.csv" 
    )