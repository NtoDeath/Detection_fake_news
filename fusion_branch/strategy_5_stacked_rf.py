"""
Stratégie 5: Stacked RandomForest Meta-Learner ⭐ ALTERNATIF PUISSANT

Logique:
  Entraîner un RandomForest meta-learner avec inputs:
    [style_pred, style_conf, knowledge_pred, knowledge_conf]
  Output: fusion prediction

Pas de gridsearch: RF entraîné sur validation subset
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from config import RANDOM_STATE


class StackedRandomForestFusion:
    """Stratégie 5: Stacked RandomForest Meta-Learner"""
    
    def __init__(self, n_estimators=100, max_depth=10):
        """
        Initialiser le meta-learner RF
        """
        self.meta_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        self.is_trained = False
    
    def train(self, style_preds, style_confs, knowledge_preds, knowledge_confs, y_true):
        """
        Entraîner le RF meta-learner sur validation subset
        
        Args:
            style_preds: (N,) array
            style_confs: (N,) array
            knowledge_preds: (N,) array
            knowledge_confs: (N,) array
            y_true: (N,) array - true labels
        """
        # Construire feature matrix
        X_meta = np.column_stack([
            style_preds,
            style_confs,
            knowledge_preds,
            knowledge_confs
        ])
        
        # Entraîner
        self.meta_model.fit(X_meta, y_true)
        self.is_trained = True
        
        print(f"✅ Stacked RF entraîné sur {len(y_true)} samples")
    
    def predict(self, style_pred, style_conf, knowledge_pred, knowledge_conf):
        """
        Inférer avec le meta-learner
        
        Args:
            style_pred: (N,) array
            style_conf: (N,) array
            knowledge_pred: (N,) array
            knowledge_conf: (N,) array
        
        Returns:
            fusion_pred: (N,) array - predictions finales
            fusion_conf: (N,) array - probabilities [0, 1]
        """
        if not self.is_trained:
            raise ValueError("Model not trained! Call train() first")
        
        # Construire feature matrix
        X_meta = np.column_stack([
            style_pred,
            style_conf,
            knowledge_pred,
            knowledge_conf
        ])
        
        # Inférer
        fusion_pred = self.meta_model.predict(X_meta)
        fusion_proba = self.meta_model.predict_proba(X_meta)
        fusion_conf = np.max(fusion_proba, axis=1)
        
        return fusion_pred, fusion_conf
    
    def get_feature_importance(self):
        """Retourner importance des features"""
        if not self.is_trained:
            raise ValueError("Model not trained!")
        
        feature_names = ['style_pred', 'style_conf', 'knowledge_pred', 'knowledge_conf']
        importances = self.meta_model.feature_importances_
        
        return dict(zip(feature_names, importances))
    
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
        
        # Feature importance
        try:
            metrics['feature_importance'] = self.get_feature_importance()
        except:
            pass
        
        return metrics
