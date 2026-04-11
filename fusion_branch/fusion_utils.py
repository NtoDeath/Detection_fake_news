"""
Fusion Analysis Utilities

Functions for evaluating fusion behavior on Part B heterogeneous dataset
and analyzing per-category performance metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')


class FusionEvaluator:
    """Evaluate fusion strategy performance on heterogeneous Part B dataset."""
    
    def __init__(self, part_B_data, style_predictions, knowledge_predictions):
        """
        Initialize evaluator.
        
        Args:
            part_B_data: DataFrame with 'text', 'label', 'category', 'reason'
            style_predictions: List of (is_fake_news, confidence) tuples from STYLE model
            knowledge_predictions: List of (verdict, confidence) tuples from KNOWLEDGE model
                verdict: 'SUPPORTED' (0), 'REFUTED' (1), 'NEUTRAL' (2)
        """
        self.part_B_data = part_B_data.reset_index(drop=True)
        self.style_preds = style_predictions
        self.knowledge_preds = knowledge_predictions
        
        # Convert predictions to DataFrames
        self.predictions_df = pd.DataFrame({
            'ground_truth': self.part_B_data['label'].values,
            'category': self.part_B_data['category'].values,
            'reason': self.part_B_data['reason'].values,
            'style_pred': [p[0] for p in style_predictions],
            'style_conf': [p[1] for p in style_predictions],
            'knowledge_pred': self._verdicts_to_labels([p[0] for p in knowledge_predictions]),
            'knowledge_conf': [p[1] for p in knowledge_predictions],
        })
    
    def _verdicts_to_labels(self, verdicts):
        """Convert verdict strings to binary labels."""
        # SUPPORTED (0) and NEUTRAL (2) -> REAL (0)
        # REFUTED (1) -> FAKE (1)
        mapping = {
            'SUPPORTED': 0,
            'REFUTED': 1,
            'NEUTRAL': 2,  # Treat NEUTRAL as uncertainty
            0: 0,
            1: 1,
            2: 2,
        }
        return [mapping.get(v, 2) for v in verdicts]
    
    def global_metrics(self, fusion_predictions):
        """
        Calculate global metrics on Part B.
        
        Args:
            fusion_predictions: List of final fusion predictions (0 or 1)
        
        Returns:
            dict with F1, Accuracy, Precision, Recall
        """
        y_true = self.predictions_df['ground_truth']
        
        metrics = {
            'accuracy': accuracy_score(y_true, fusion_predictions),
            'f1': f1_score(y_true, fusion_predictions),
            'precision': precision_score(y_true, fusion_predictions),
            'recall': recall_score(y_true, fusion_predictions),
        }
        
        # Per-class metrics
        report = classification_report(y_true, fusion_predictions, output_dict=True)
        metrics['real_f1'] = report['0']['f1-score']
        metrics['fake_f1'] = report['1']['f1-score']
        
        return metrics
    
    def per_category_metrics(self, fusion_predictions):
        """
        Evaluate fusion performance per category.
        
        Answers: For each case type, does fusion handle it correctly?
        """
        results = {}
        
        for category in self.part_B_data['category'].unique():
            mask = self.predictions_df['category'] == category
            y_true_cat = self.predictions_df.loc[mask, 'ground_truth']
            y_pred_cat = np.array(fusion_predictions)[mask]
            
            if len(y_true_cat) == 0:
                continue
            
            results[category] = {
                'samples': len(y_true_cat),
                'accuracy': accuracy_score(y_true_cat, y_pred_cat),
                'f1': f1_score(y_true_cat, y_pred_cat, zero_division=0),
                'precision': precision_score(y_true_cat, y_pred_cat, zero_division=0),
                'recall': recall_score(y_true_cat, y_pred_cat, zero_division=0),
                'fake_ratio': (y_true_cat == 1).sum() / len(y_true_cat),
            }
        
        return pd.DataFrame(results).T
    
    def model_agreement_analysis(self):
        """
        Analyze agreement between STYLE and KNOWLEDGE models.
        
        Returns:
            dict with agreement metrics per category
        """
        self.predictions_df['agree'] = (
            self.predictions_df['style_pred'] == 
            (self.predictions_df['knowledge_pred'] < 2)  # NEUTRAL (2) is uncertain
        )
        
        results = {}
        
        for category in self.part_B_data['category'].unique():
            mask = self.predictions_df['category'] == category
            subset = self.predictions_df[mask]
            
            results[category] = {
                'total': len(subset),
                'agreement_rate': subset['agree'].sum() / len(subset),
                'avg_style_conf': subset['style_conf'].mean(),
                'avg_knowledge_conf': subset['knowledge_conf'].mean(),
                'style_only_fake': ((subset['style_pred'] == 1) & (subset['knowledge_pred'] != 1)).sum(),
                'knowledge_only_fake': ((subset['knowledge_pred'] == 1) & (subset['style_pred'] != 1)).sum(),
            }
        
        return pd.DataFrame(results).T
    
    def confidence_analysis(self):
        """
        Analyze confidence distribution patterns.
        
        Identifies confidence threshold sweet spots.
        """
        results = {}
        
        for category in self.part_B_data['category'].unique():
            mask = self.predictions_df['category'] == category
            subset = self.predictions_df[mask]
            
            # Correct predictions: where fusion got it right
            correct = subset[subset['style_pred'] == subset['ground_truth']]
            incorrect = subset[subset['style_pred'] != subset['ground_truth']]
            
            results[category] = {
                'correct_style_conf_mean': correct['style_conf'].mean() if len(correct) > 0 else 0,
                'incorrect_style_conf_mean': incorrect['style_conf'].mean() if len(incorrect) > 0 else 0,
                'correct_knowledge_conf_mean': correct['knowledge_conf'].mean() if len(correct) > 0 else 0,
                'incorrect_knowledge_conf_mean': incorrect['knowledge_conf'].mean() if len(incorrect) > 0 else 0,
            }
        
        return pd.DataFrame(results).T
    
    def plot_category_performance(self, fusion_predictions, figsize=(14, 6)):
        """Visualize per-category performance."""
        metrics_df = self.per_category_metrics(fusion_predictions)
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Per-category F1
        metrics_df['f1'].sort_values().plot(
            kind='barh', ax=axes[0], color='steelblue'
        )
        axes[0].set_title('F1 Score by Category')
        axes[0].set_xlabel('F1 Score')
        axes[0].axvline(x=metrics_df['f1'].mean(), color='red', linestyle='--', label='Mean')
        axes[0].legend()
        
        # Sample count per category
        metrics_df['samples'].sort_values().plot(
            kind='barh', ax=axes[1], color='coral'
        )
        axes[1].set_title('Sample Count by Category')
        axes[1].set_xlabel('Number of Samples')
        
        plt.tight_layout()
        return fig
    
    def plot_agreement_heatmap(self, figsize=(10, 6)):
        """Visualize model agreement patterns per category."""
        agreement_df = self.model_agreement_analysis()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap data: rows=categories, cols=metrics
        heatmap_data = agreement_df[[
            'agreement_rate',
            'avg_style_conf',
            'avg_knowledge_conf'
        ]]
        
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax)
        ax.set_title('Model Agreement & Confidence Patterns by Category')
        plt.tight_layout()
        return fig
    
    def print_summary(self, fusion_predictions):
        """Print comprehensive evaluation summary."""
        print("\n" + "="*70)
        print("FUSION EVALUATION SUMMARY (Part B Heterogeneous Dataset)")
        print("="*70)
        
        # Global metrics
        metrics = self.global_metrics(fusion_predictions)
        print("\n[GLOBAL PERFORMANCE]")
        print(f"  Accuracy : {metrics['accuracy']:.4f}")
        print(f"  F1 Score : {metrics['f1']:.4f} (FAKE: {metrics['fake_f1']:.4f}, REAL: {metrics['real_f1']:.4f})")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall   : {metrics['recall']:.4f}")
        
        # Per-category performance
        print("\n[PER-CATEGORY PERFORMANCE]")
        cat_metrics = self.per_category_metrics(fusion_predictions)
        print(cat_metrics.to_string())
        
        # Model agreement
        print("\n[MODEL AGREEMENT ANALYSIS]")
        agreement_df = self.model_agreement_analysis()
        print(agreement_df.to_string())
        
        # Confidence patterns
        print("\n[CONFIDENCE PATTERNS]")
        conf_df = self.confidence_analysis()
        print(conf_df.to_string())
        
        print("\n" + "="*70)


class ThresholdOptimizer:
    """
    Optimize fusion thresholds using evaluator results.
    """
    
    def __init__(self, evaluator):
        self.evaluator = evaluator
    
    def grid_search(self, threshold_ranges, metric='f1', verbose=True):
        """
        Grid search over threshold ranges.
        
        Args:
            threshold_ranges: dict with 'confident', 'uncertain', 'strong_confident' keys
                Example: {
                    'confident': np.linspace(0.5, 0.8, 7),
                    'uncertain': np.linspace(0.3, 0.6, 7),
                    'strong_confident': np.linspace(0.7, 0.95, 6),
                }
            metric: 'f1', 'accuracy', etc.
            verbose: print progress
        
        Returns:
            DataFrame with results, sorted by metric (best first)
        """
        results = []
        
        from itertools import product
        
        configs = product(
            threshold_ranges.get('confident', [0.65]),
            threshold_ranges.get('uncertain', [0.50]),
            threshold_ranges.get('strong_confident', [0.85]),
        )
        
        total = (
            len(threshold_ranges.get('confident', [1])) *
            len(threshold_ranges.get('uncertain', [1])) *
            len(threshold_ranges.get('strong_confident', [1]))
        )
        
        for i, (conf, unc, strong) in enumerate(configs):
            if verbose and i % max(1, total // 10) == 0:
                print(f"  Progress: {i}/{total}")
            
            # Simulate fusion with these thresholds
            fusion_preds = self._apply_thresholds(conf, unc, strong)
            metrics = self.evaluator.global_metrics(fusion_preds)
            
            results.append({
                'confident_threshold': conf,
                'uncertain_threshold': unc,
                'strong_confident_threshold': strong,
                'f1': metrics['f1'],
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
            })
        
        results_df = pd.DataFrame(results)
        return results_df.sort_values(metric, ascending=False)
    
    def _apply_thresholds(self, conf_th, unc_th, strong_th):
        """Apply thresholds to generate fusion predictions."""
        # Simplified fusion: majority voting with confidence-based fallback
        predictions = []
        
        for _, row in self.evaluator.predictions_df.iterrows():
            style_pred = row['style_pred']
            style_conf = row['style_conf']
            knowledge_pred = row['knowledge_pred']
            knowledge_conf = row['knowledge_conf']
            
            # Check confidence levels
            style_confident = style_conf >= conf_th
            knowledge_confident = knowledge_conf >= conf_th
            
            if style_confident and knowledge_confident:
                # Both confident: majority voting
                if style_pred == knowledge_pred:
                    # Agreement
                    pred = style_pred
                else:
                    # Disagreement: use higher confidence
                    pred = style_pred if style_conf >= knowledge_conf else knowledge_pred
            elif style_confident:
                # Only style confident
                pred = style_pred
            elif knowledge_confident:
                # Only knowledge confident
                pred = knowledge_pred
            else:
                # Both uncertain: use uncertainty threshold
                combined_conf = (style_conf + knowledge_conf) / 2
                if combined_conf >= unc_th:
                    pred = style_pred  # Default to style
                else:
                    pred = 1  # Default to FAKE when very uncertain
            
            predictions.append(pred)
        
        return predictions


if __name__ == '__main__':
    print("Fusion utilities module. Import and use FusionEvaluator class.")
