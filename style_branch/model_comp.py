import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import joblib
from contextlib import contextmanager
import time
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, f1_score, log_loss

print("Loading data (Style + RoBERTa Semantics)...")
df_train = pd.read_csv("../data/block_B_train_WITH_PROB.csv")
df_test = pd.read_csv("../data/block_C_final_test_WITH_PROB.csv")

columns_to_ignore = ['label', 'text']

X_train = df_train.drop(columns=[col for col in columns_to_ignore if col in df_train.columns])
y_train = df_train['label']

X_test = df_test.drop(columns=[col for col in columns_to_ignore if col in df_test.columns])
y_test = df_test['label']

N_ITERATIONS = 15 
total_tasks = N_ITERATIONS * 5

param_grid_rf = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': ['balanced', None]
}

param_grid_xgb = {
    'n_estimators': [100, 200, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 10],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
}

print(f"\nConfiguration: {N_ITERATIONS} combinations tested per model (Cross-Validation x5).")


@contextmanager
def tqdm_joblib(tqdm_object):
    """Patch to link joblib and tqdm in order to display a progress bar"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


# RANDOM FOREST
print("Optimizing Random Forest...\n")
start_time = time.time()

rf_base = RandomForestClassifier(random_state=42)
search_rf = RandomizedSearchCV(
    estimator=rf_base, param_distributions=param_grid_rf, 
    n_iter=N_ITERATIONS, scoring='roc_auc', cv=5, 
    n_jobs=-1, random_state=42, verbose=0
)
with tqdm_joblib(tqdm(desc="Training Random Forest", total=total_tasks)):
    search_rf.fit(X_train, y_train)

best_rf = search_rf.best_estimator_
rf_time = round((time.time() - start_time) / 60, 2)
print(f"Finished in {rf_time} min. Best parameters: {search_rf.best_params_}")

# XGBOOST
print("Optimizing XGBoost...\n")
start_time = time.time()

xgb_base = XGBClassifier(random_state=42, eval_metric='logloss')
search_xgb = RandomizedSearchCV(
    estimator=xgb_base, param_distributions=param_grid_xgb, 
    n_iter=N_ITERATIONS, scoring='roc_auc', cv=5, 
    n_jobs=-1, random_state=42, verbose=0
)
with tqdm_joblib(tqdm(desc="Training XGBoost", total=total_tasks, colour='green')):
    search_xgb.fit(X_train, y_train)

best_xgb = search_xgb.best_estimator_
xgb_time = round((time.time() - start_time) / 60, 2)
print(f"Finished in {xgb_time} min. Best parameters: {search_xgb.best_params_}\n")

# RF Evaluation
pred_rf = best_rf.predict(X_test)
proba_rf = best_rf.predict_proba(X_test)[:, 1]
acc_rf = accuracy_score(y_test, pred_rf)
roc_rf = roc_auc_score(y_test, proba_rf)
f1_rf = f1_score(y_test, pred_rf)
log_loss_rf = log_loss(y_test, proba_rf)

# XGB Evaluation
pred_xgb = best_xgb.predict(X_test)
proba_xgb = best_xgb.predict_proba(X_test)[:, 1]
acc_xgb = accuracy_score(y_test, pred_xgb)
roc_xgb = roc_auc_score(y_test, proba_xgb)
f1_xgb = f1_score(y_test, pred_xgb)
log_loss_xgb = log_loss(y_test, proba_xgb)

print(f"{'Model':<20} | {'Accuracy':<20} | {'ROC-AUC (Quality)':<20} | {'F1 Score':<20} | {'Log Loss':<20} |")
print("-" * 114)
print(f"{'Random Forest':<20} | {acc_rf*100:>19.2f}% | {roc_rf*100:>19.2f}% | {f1_rf*100:>19.2f}% | {log_loss_rf*100:>19.2f}% |")
print(f"{'XGBoost':<20} | {acc_xgb*100:>19.2f}% | {roc_xgb*100:>19.2f}% | {f1_xgb*100:>19.2f}% | {log_loss_xgb*100:>19.2f}% |")
print("-" * 114)

if f1_xgb > f1_rf:
    winner, winner_name, winner_predictions = best_xgb, "XGBoost", pred_xgb
elif f1_rf > f1_xgb:
    winner, winner_name, winner_predictions = best_rf, "Random Forest", pred_rf
else:
    if log_loss_xgb < log_loss_rf:
        winner, winner_name, winner_predictions = best_xgb, "XGBoost", pred_xgb
    else:
        winner, winner_name, winner_predictions = best_rf, "Random Forest", pred_rf

print(f"\n The best model is: {winner_name.upper()}\n")
print("Detailed report of the best model:")
report = classification_report(y_test, winner_predictions)
print(report)

model_filename = f"best_model.pkl"
joblib.dump(winner, f"results/{model_filename}")
print(f"The best model has been saved as: {model_filename}")

report_filename = f"report_{winner_name.lower().replace(' ', '_')}.txt"
with open(f"results/{report_filename}", "w") as f:
    f.write(f"Model: {winner_name}\n")
    f.write(report)
print(f"Performance report saved to: results/{report_filename}")

importances = winner.feature_importances_
feature_names = X_train.columns

df_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
df_importances = df_importances.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=df_importances, palette='viridis', hue='Feature', legend=False)
plt.title(f"Which features convinced {winner_name}? (Semantics vs Style)", fontsize=14)
plt.xlabel("Weight in decision")
plt.ylabel("Features")
plt.tight_layout()

image_filename = f"feature_weights_{winner_name.lower().replace(' ', '_')}.png"
plt.savefig(f"results/{image_filename}")
print(f"Graph generated: results/{image_filename}")