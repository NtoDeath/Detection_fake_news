"""
Stratégie 2: Confidence-Weighted Voting

Logique:
  Poids = Performance histórique des modèles (Style 92%, Knowledge 32%)
  vote = (w_style * style_conf) * style_pred + (w_knowledge * knowledge_conf) * knowledge_pred
  prediction = (vote >= 0.5) → 1 else 0

Pas de gridsearch: poids fixes basés sur performance Part A
"""

import numpy as np
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from config import STYLE_BASELINE_F1, KNOWLEDGE_BASELINE_F1


class ConfidenceWeightedVoting:
    """Stratégie 2: Confidence-Weighted Voting"""
    
    def __init__(self, style_baseline=STYLE_BASELINE_F1, knowledge_baseline=KNOWLEDGE_BASELINE_F1):
        """
        Initialiser avec poids basés sur baselines
        """
        total = style_baseline + knowledge_baseline
        self.w_style = style_baseline / total
        self.w_knowledge = knowledge_baseline / total
    
    def predict(self, style_pred, style_conf, knowledge_pred, knowledge_conf):
        """
        Faire prédiction avec poids adaptatifs
        
        Args:
            style_pred: (N,) array - predictions binaires Style
            style_conf: (N,) array - confidence scores Style [0, 1]
            knowledge_pred: (N,) array - predictions binaires Knowledge
            knowledge_conf: (N,) array - confidence scores Knowledge [0, 1]
        
        Returns:
            fusion_pred: (N,) array - predictions finales
            fusion_conf: (N,) array - confidence finales
        """
        # Poids adaptatifs: baseline * confidence
        adaptive_w_style = self.w_style * style_conf
        adaptive_w_knowledge = self.w_knowledge * knowledge_conf
        
        # Normaliser les poids
        total_w = adaptive_w_style + adaptive_w_knowledge + 1e-7
        norm_w_style = adaptive_w_style / total_w
        norm_w_knowledge = adaptive_w_knowledge / total_w
        
        # Vote pondéré
        fusion_score = norm_w_style * style_pred + norm_w_knowledge * knowledge_pred
        fusion_pred = (fusion_score >= 0.5).astype(int)
        fusion_conf = np.abs(fusion_score - 0.5) * 2  # Distance from boundary
        fusion_conf = np.clip(fusion_conf, 0, 1)
        
        return fusion_pred, fusion_conf
    
    def evaluate(self, style_preds, style_confs, knowledge_preds, knowledge_confs, y_true):
        """Évaluer les performances"""
        preds, confs = self.predict(style_preds, style_confs, knowledge_preds, knowledge_confs)
        
        metrics = {
            'accuracy': accuracy_score(y_true, preds),
            'precision': precision_score(y_true, preds),
            'recall': recall_score(y_true, preds),
            'f1': f1_score(y_true, preds),
            'roc_auc': roc_auc_score(y_true, confs),
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_true, preds)
        tn, fp, fn, tp = cm.ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['confusion_matrix'] = {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
        
        return metrics
