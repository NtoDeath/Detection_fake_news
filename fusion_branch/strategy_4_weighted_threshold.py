"""
Stratégie 4: Weighted Voting + Threshold Optimization ⭐ MEILLEUR

Logique:
  vote = w_style * style_pred + w_knowledge * knowledge_pred
  prediction = (vote >= threshold) → 1 else 0

Gridsearch: 5 × 5 × 5 = 125 configurations
  w_style ∈ [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
  w_knowledge ∈ [0.1, 0.2, 0.3, 0.4, 0.5]
  threshold ∈ [0.3, 0.4, 0.5, 0.6, 0.7]
"""

import numpy as np
from itertools import product
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix


class WeightedVotingWithThreshold:
    """Stratégie 4: Weighted Voting + Threshold ⭐ MEILLEUR"""
    
    def __init__(self, w_style=0.65, w_knowledge=0.35, threshold=0.50):
        """
        Initialiser avec poids et seuil
        
        Args:
            w_style: poids Style ∈ [0, 1]
            w_knowledge: poids Knowledge ∈ [0, 1]
            threshold: seuil de décision ∈ [0, 1]
        """
        self.w_style = w_style
        self.w_knowledge = w_knowledge
        self.threshold = threshold
    
    def predict(self, style_pred, style_conf, knowledge_pred, knowledge_conf):
        """
        Faire prédiction avec vote pondéré
        
        Args:
            style_pred: (N,) array - predictions binaires Style
            style_conf: (N,) array - confidence scores Style [0, 1]  (ignoré)
            knowledge_pred: (N,) array - predictions binaires Knowledge
            knowledge_conf: (N,) array - confidence scores Knowledge [0, 1] (ignoré)
        
        Returns:
            fusion_pred: (N,) array - predictions finales
            fusion_conf: (N,) array - confidence finales
        """
        # Vote pondéré (ignorer confiance pour stratégie simple)
        fusion_score = self.w_style * style_pred + self.w_knowledge * knowledge_pred
        fusion_pred = (fusion_score >= self.threshold).astype(int)
        
        # Confiance: distance from threshold
        fusion_conf = np.abs(fusion_score - self.threshold)
        
        return fusion_pred, fusion_conf
    
    def gridsearch_params(self, style_preds, style_confs, knowledge_preds, knowledge_confs, y_true, verbose=False):
        """
        Gridsearch pour trouver meilleur (w_style, w_knowledge, threshold)
        
        Returns:
            best_params: dict avec w_style, w_knowledge, threshold
            best_f1: float
            results: list of all configurations tested
        """
        best_f1 = -1
        best_params = {}
        results = []
        
        # Gridsearch all combinations
        w_styles = np.arange(0.4, 0.95, 0.1)
        w_knowledges = np.arange(0.1, 0.6, 0.1)
        thresholds = np.arange(0.3, 0.8, 0.1)
        
        total_configs = len(w_styles) * len(w_knowledges) * len(thresholds)
        config_idx = 0
        
        for w_style, w_knowledge, threshold in product(w_styles, w_knowledges, thresholds):
            config_idx += 1
            
            self.w_style = w_style
            self.w_knowledge = w_knowledge
            self.threshold = threshold
            
            preds, _ = self.predict(style_preds, style_confs, knowledge_preds, knowledge_confs)
            
            try:
                f1 = f1_score(y_true, preds)
                acc = accuracy_score(y_true, preds)
            except:
                f1 = 0
                acc = 0
            
            results.append({
                'w_style': w_style,
                'w_knowledge': w_knowledge,
                'threshold': threshold,
                'f1': f1,
                'accuracy': acc,
            })
            
            if f1 > best_f1:
                best_f1 = f1
                best_params = {
                    'w_style': w_style,
                    'w_knowledge': w_knowledge,
                    'threshold': threshold,
                }
            
            if verbose and config_idx % 25 == 0:
                print(f"  [{config_idx}/{total_configs}] Best F1: {best_f1:.4f}")
        
        # Set best params
        self.w_style = best_params['w_style']
        self.w_knowledge = best_params['w_knowledge']
        self.threshold = best_params['threshold']
        
        return best_params, best_f1, results
    
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
