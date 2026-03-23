"""Hyperparameter tuning and ensemble comparison (RandomForest vs XGBoost).

Trains RandomForest and XGBoost classifiers on augmented dataset
(style features + RoBERTa probabilities). Uses RandomizedSearchCV
to optimize hyperparameters on validation set (Block B).

Ensemble Strategy
-----------------
Input: 20+ style metrics + 1 RoBERTa super-feature (total ~21 features)
Models: Random Forest (tree bagging) and XGBoost (gradient boosting)
Evaluation: 5-fold cross-validation with 15 random configurations each
Winner: Selected by F1 score (primary), then log loss (tiebreaker)

Output Files
------------
- results/best_model.pkl: Serialized winning model (scikit-learn format)
- results/report_{model}.txt: Classification report per fold
- results/feature_weights_{model}.png: Feature importance barplot

Dependencies
------------
- xgboost: GBM classifier
- sklearn: RandomForest + metrics + CV
- matplotlib/seaborn: feature visualization
- joblib: parallel training + model serialization
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Generator, Tuple
from tqdm.auto import tqdm
import joblib
from contextlib import contextmanager
import time
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, f1_score, log_loss

print("Loading data (Style + RoBERTa Semantics)...")
df_train: pd.DataFrame = pd.read_csv("../data/block_B_train_WITH_PROB.csv")
df_test: pd.DataFrame = pd.read_csv("../data/block_C_final_test_WITH_PROB.csv")

columns_to_ignore: list = ['label', 'text']

# Prepare training set (features only, excluding text and label)
X_train: pd.DataFrame = df_train.drop(columns=[col for col in columns_to_ignore if col in df_train.columns])
y_train: pd.Series = df_train['label']

# Prepare test set (same transformations)
X_test: pd.DataFrame = df_test.drop(columns=[col for col in columns_to_ignore if col in df_test.columns])
y_test: pd.Series = df_test['label']

# Hyperparameter search configuration
N_ITERATIONS: int = 15  # Random configurations to test
total_tasks: int = N_ITERATIONS * 5  # 5-fold cross-validation

# Random Forest hyperparameter grid
param_grid_rf: Dict[str, Any] = {
    'n_estimators': [100, 200, 300, 500],  # Number of trees
    'max_depth': [10, 20, 30, None],  # Tree depth (None = unlimited)
    'min_samples_split': [2, 5, 10],  # Min samples to split node
    'min_samples_leaf': [1, 2, 4],  # Min samples in leaf node
    'class_weight': ['balanced', None]  # Auto-balance class imbalance
}

# XGBoost hyperparameter grid
param_grid_xgb: Dict[str, Any] = {
    'n_estimators': [100, 200, 300, 500],  # Boosting rounds
    'learning_rate': [0.01, 0.05, 0.1, 0.2],  # Shrinkage parameter
    'max_depth': [3, 5, 7, 10],  # Tree depth (smaller = simpler)
    'subsample': [0.7, 0.8, 0.9, 1.0],  # Row sampling for robustness
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0]  # Feature sampling per tree
}

print(f"\nConfiguration: {N_ITERATIONS} combinations tested per model (Cross-Validation x5).")


@contextmanager
def tqdm_joblib(tqdm_object: tqdm) -> Generator:
    """Context manager to integrate tqdm progress bar with joblib parallelization.
    
    Patches joblib's batch callback to update tqdm after each completed batch.
    Restores original callback on exit.
    
    Parameters
    ----------
    tqdm_object : tqdm
        Progress bar object to update during joblib execution.
    
    Yields
    ------
    tqdm
        The input tqdm object for use within the with block.
    
    Notes
    -----
    Replaces joblib.parallel.BatchCompletionCallBack temporarily.
    """
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            # Update progress bar with the number of completed samples
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    # Temporarily replace joblib's completion callback
    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        # Restore original callback
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


# RANDOM FOREST
print("Optimizing Random Forest...\n")
start_time: float = time.time()

# Initialize base Random Forest classifier
rf_base: RandomForestClassifier = RandomForestClassifier(random_state=42)

# Set up RandomizedSearchCV for hyperparameter tuning
# ROC-AUC scoring optimizes for probability calibration (important for fake news detection)
search_rf: RandomizedSearchCV = RandomizedSearchCV(
    estimator=rf_base, param_distributions=param_grid_rf, 
    n_iter=N_ITERATIONS,  # Test 15 random combinations
    scoring='roc_auc',  # Metric for hyperparameter optimization
    cv=5,  # 5-fold cross-validation
    n_jobs=-1,  # Use all available CPU cores
    random_state=42, verbose=0
)

# Fit with progress bar integration
with tqdm_joblib(tqdm(desc="Training Random Forest", total=total_tasks)):
    search_rf.fit(X_train, y_train)

best_rf: RandomForestClassifier = search_rf.best_estimator_
rf_time: float = round((time.time() - start_time) / 60, 2)
print(f"Finished in {rf_time} min. Best parameters: {search_rf.best_params_}")

# XGBOOST
print("Optimizing XGBoost...\n")
start_time: float = time.time()

# Initialize base XGBoost classifier with logloss evaluation metric
xgb_base: XGBClassifier = XGBClassifier(random_state=42, eval_metric='logloss')

# Set up RandomizedSearchCV for hyperparameter tuning
# Same CV strategy as Random Forest for fair comparison
search_xgb: RandomizedSearchCV = RandomizedSearchCV(
    estimator=xgb_base, param_distributions=param_grid_xgb, 
    n_iter=N_ITERATIONS,  # Test 15 random combinations
    scoring='roc_auc',  # Metric for hyperparameter optimization
    cv=5,  # 5-fold cross-validation
    n_jobs=-1,  # Use all available CPU cores
    random_state=42, verbose=0
)

# Fit with progress bar integration (green color for distinction)
with tqdm_joblib(tqdm(desc="Training XGBoost", total=total_tasks, colour='green')):
    search_xgb.fit(X_train, y_train)

best_xgb: XGBClassifier = search_xgb.best_estimator_
xgb_time: float = round((time.time() - start_time) / 60, 2)
print(f"Finished in {xgb_time} min. Best parameters: {search_xgb.best_params_}\n")

# Evaluate Random Forest on test set
# Prediction: hard labels (0/1)
# Probabilities: confidence for class 1 (fake news)
pred_rf: 'np.ndarray' = best_rf.predict(X_test)
proba_rf: 'np.ndarray' = best_rf.predict_proba(X_test)[:, 1]
acc_rf: float = accuracy_score(y_test, pred_rf)
roc_rf: float = roc_auc_score(y_test, proba_rf)
f1_rf: float = f1_score(y_test, pred_rf)
log_loss_rf: float = log_loss(y_test, proba_rf)

# Evaluate XGBoost on test set (same metrics)
pred_xgb: 'np.ndarray' = best_xgb.predict(X_test)
proba_xgb: 'np.ndarray' = best_xgb.predict_proba(X_test)[:, 1]
acc_xgb: float = accuracy_score(y_test, pred_xgb)
roc_xgb: float = roc_auc_score(y_test, proba_xgb)
f1_xgb: float = f1_score(y_test, pred_xgb)
log_loss_xgb: float = log_loss(y_test, proba_xgb)

print(f"{'Model':<20} | {'Accuracy':<20} | {'ROC-AUC (Quality)':<20} | {'F1 Score':<20} | {'Log Loss':<20} |")
print("-" * 114)
print(f"{'Random Forest':<20} | {acc_rf*100:>19.2f}% | {roc_rf*100:>19.2f}% | {f1_rf*100:>19.2f}% | {log_loss_rf*100:>19.2f}% |")
print(f"{'XGBoost':<20} | {acc_xgb*100:>19.2f}% | {roc_xgb*100:>19.2f}% | {f1_xgb*100:>19.2f}% | {log_loss_xgb*100:>19.2f}% |")
print("-" * 114)

# Winner selection logic: F1 as primary criterion, log loss as tiebreaker
if f1_xgb > f1_rf:
    # XGBoost better at balancing precision-recall
    winner: 'RandomForestClassifier | XGBClassifier' = best_xgb
    winner_name: str = "XGBoost"
    winner_predictions: 'np.ndarray' = pred_xgb
elif f1_rf > f1_xgb:
    # Random Forest better at balancing precision-recall
    winner: 'RandomForestClassifier | XGBClassifier' = best_rf
    winner_name: str = "Random Forest"
    winner_predictions: 'np.ndarray' = pred_rf
else:
    # F1 scores equal: break tie with log loss (lower is better)
    if log_loss_xgb < log_loss_rf:
        winner: 'RandomForestClassifier | XGBClassifier' = best_xgb
        winner_name: str = "XGBoost"
        winner_predictions: 'np.ndarray' = pred_xgb
    else:
        winner: 'RandomForestClassifier | XGBClassifier' = best_rf
        winner_name: str = "Random Forest"
        winner_predictions: 'np.ndarray' = pred_rf

print(f"\n The best model is: {winner_name.upper()}\n")
print("Detailed report of the best model:")
report: str = classification_report(y_test, winner_predictions)
print(report)

# Save trained model for inference
model_filename: str = f"best_model.pkl"
joblib.dump(winner, f"results/{model_filename}")
print(f"The best model has been saved as: {model_filename}")

# Save classification report to text file
report_filename: str = f"report_{winner_name.lower().replace(' ', '_')}.txt"
with open(f"results/{report_filename}", "w") as f:
    f.write(f"Model: {winner_name}\n")
    f.write(report)
print(f"Performance report saved to: results/{report_filename}")

# Feature importance analysis: identify which features drive predictions
importances: 'np.ndarray' = winner.feature_importances_
feature_names: 'Index' = X_train.columns

# Create DataFrame for visualization
df_importances: pd.DataFrame = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
df_importances = df_importances.sort_values(by='Importance', ascending=False)

# Barplot: top contributing features
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=df_importances, palette='viridis', hue='Feature', legend=False)
plt.title(f"Which features convinced {winner_name}? (Semantics vs Style)", fontsize=14)
plt.xlabel("Weight in decision")
plt.ylabel("Features")
plt.tight_layout()

# Save feature importance plot
image_filename: str = f"feature_weights_{winner_name.lower().replace(' ', '_')}.png"
plt.savefig(f"results/{image_filename}")
print(f"Graph generated: results/{image_filename}")