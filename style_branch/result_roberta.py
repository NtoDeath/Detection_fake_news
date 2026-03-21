import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss, classification_report
from scipy.special import softmax
from tqdm import tqdm # Pour avoir une barre de progression

def evaluate_roberta_baseline(model_path="./roberta_fine_tunned", test_data_path = "../data/block_C_final_test_WITH_PROB.csv"):    
    device = torch.device("mps" if torch.backends.mps.is_available() else "gpu" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    df_test = pd.read_csv(test_data_path)
    textes = df_test['text'].tolist()
    labels_vrais = df_test['label'].tolist()
    
    predictions = []
    probabilites_classe_1 = []

    batch_size = 16 
    
    with torch.no_grad():
        for i in tqdm(range(0, len(textes), batch_size), desc="Évaluation RoBERTa"):
            batch_textes = textes[i:i + batch_size]
            
            inputs = tokenizer(batch_textes, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            logits = outputs.logits.detach().cpu().numpy()
            
            probs = softmax(logits, axis=1)
            preds = logits.argmax(axis=-1)
            
            predictions.extend(preds)
            probabilites_classe_1.extend(probs[:, 1])

    acc = accuracy_score(labels_vrais, predictions)
    f1 = f1_score(labels_vrais, predictions)
    roc_auc = roc_auc_score(labels_vrais, probabilites_classe_1)
    ll = log_loss(labels_vrais, probabilites_classe_1)

    print("\n" + "="*50)
    print("RÉSULTATS DE ROBERTA SEUL (BASELINE)")
    print("="*50)
    print(f"Accuracy  : {acc * 100:.2f}%")
    print(f"F1 Score  : {f1 * 100:.2f}%")
    print(f"ROC-AUC   : {roc_auc * 100:.2f}%")
    print(f"Log Loss  : {ll * 100:.2f}%")
    print("-" * 50)
    print("\nRapport détaillé :\n")
    print(classification_report(labels_vrais, predictions))

if __name__ == "__main__":
    # Remplacez "test_data.csv" par le chemin exact de vos 12335 lignes de test
    evaluate_roberta_baseline(
        model_path="./roberta_fine_tunned", 
        test_data_path="../data/block_C_final_test_WITH_PROB.csv" 
    )