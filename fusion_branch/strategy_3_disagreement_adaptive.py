"""
Stratégie 3: Disagreement-Adaptive Weighting

Logique:
  Si accord (prédictions identiques):
    → Confiance haute (moyenne des confiances * 1.2)
  Si désaccord:
    → Vote égal (0.5 chacun)

Optimisation: Gridsearch sur disagreement_weight
"""

import numpy as np
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix


class DisagreementAdaptiveWeighting:
    """Stratégie 3: Disagreement-Adaptive Weighting"""
    
    def __init__(self, disagreement_weight=0.5):
        self.disagreement_weight = disagreement_weight
    
    def predict(self, style_pred, style_conf, knowledge_pred, knowledge_conf):
        """
        Adapter les poids basé sur accord/désaccord
        
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
        
        # Détecter accord/désaccord
        agreement = (style_pred == knowledge_pred)
        
        # Cas d'accord: confiance haute
        agree_mask = agreement
        if np.any(agree_mask):
            # Poids: donner advantage au modèle confiant
            best_conf = np.where(style_conf[agree_mask] > knowledge_conf[agree_mask], 
                                 style_conf[agree_mask], 
                                 knowledge_conf[agree_mask])
            fusion_pred[agree_mask] = style_pred[agree_mask]  # Ils s'accordent donc c'est la même
            fusion_conf[agree_mask] = np.clip(best_conf * 1.2, 0, 1)
        
        # Cas de désaccord: vote égal
        disagree_mask = ~agreement
        if np.any(disagree_mask):
            vote = 0.5 * style_pred[disagree_mask] + 0.5 * knowledge_pred[disagree_mask]
            fusion_pred[disagree_mask] = (vote >= 0.5).astype(int)
            # Confiance: moyenne mit un discount pour désaccord
            fusion_conf[disagree_mask] = (style_conf[disagree_mask] + knowledge_conf[disagree_mask]) / 2 * (1 - self.disagreement_weight / 10)
        
        return fusion_pred, fusion_conf
    
    def gridsearch_params(self, style_preds, style_confs, knowledge_preds, knowledge_confs, y_true):
        """
        Gridsearch pour trouver le meilleur disagreement_weight
        """
        best_f1 = -1
        best_weight = 0.5
        results = []
        
        for weight in np.arange(0.3, 0.9, 0.1):
            self.disagreement_weight = weight
            preds, _ = self.predict(style_preds, style_confs, knowledge_preds, knowledge_confs)
            
            f1 = f1_score(y_true, preds)
            results.append({'disagreement_weight': weight, 'f1': f1})
            
            if f1 > best_f1:
                best_f1 = f1
                best_weight = weight
        
        self.disagreement_weight = best_weight
        return {'disagreement_weight': best_weight}, best_f1, results
    
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
