"""
Stratégie 1: Cascading (Style First)

Logique: 
  - Si Style confiant (conf >= threshold) → utiliser Style
  - Sinon → utiliser Knowledge

Optimisation: Gridsearch sur threshold ∈ [0.5, 0.55, 0.6, ..., 0.9]
"""

import numpy as np
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix


class CascadingStyleFirst:
    """Stratégie 1: Cascading avec Style dominant"""
    
    def __init__(self, style_threshold=0.85):
        self.style_threshold = style_threshold
    
    def predict(self, style_pred, style_conf, knowledge_pred, knowledge_conf):
        """
        Faire prédiction basée sur confiance Style
        
        Args:
            style_pred: (N,) array - predictions binaires Style
            style_conf: (N,) array - confidence scores Style [0, 1]
            knowledge_pred: (N,) array - predictions binaires Knowledge
            knowledge_conf: (N,) array - confidence scores Knowledge [0, 1]
        
        Returns:
            fusion_pred: (N,) array - predictions finales
            fusion_conf: (N,) array - confidence finales
        """
        fusion_pred = np.zeros_like(style_pred, dtype=int)
        fusion_conf = np.zeros_like(style_conf, dtype=float)
        
        # Appliquer logique de cascade
        high_conf = style_conf >= self.style_threshold
        fusion_pred[high_conf] = style_pred[high_conf]
        fusion_conf[high_conf] = style_conf[high_conf]
        
        # Si Style pas confiant → utiliser Knowledge
        low_conf = ~high_conf
        fusion_pred[low_conf] = knowledge_pred[low_conf]
        fusion_conf[low_conf] = knowledge_conf[low_conf]
        
        return fusion_pred, fusion_conf
    
    def gridsearch_params(self, style_preds, style_confs, knowledge_preds, knowledge_confs, y_true):
        """
        Gridsearch pour trouver le meilleur threshold
        
        Args:
            y_true: (N,) array - true labels
        
        Returns:
            best_threshold: float
            best_f1: float
        """
        best_f1 = -1
        best_threshold = 0.5
        results = []
        
        for threshold in np.arange(0.5, 0.95, 0.05):
            self.style_threshold = threshold
            preds, _ = self.predict(style_preds, style_confs, knowledge_preds, knowledge_confs)
            
            f1 = f1_score(y_true, preds)
            results.append({'threshold': threshold, 'f1': f1})
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        self.style_threshold = best_threshold
        return {'threshold': best_threshold}, best_f1, results
    
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
