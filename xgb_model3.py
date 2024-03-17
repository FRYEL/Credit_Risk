import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from skopt import BayesSearchCV
import mlflow
import mlflow.xgboost
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

df = pd.read_csv('data/cleaned_data.csv', low_memory=False)

# Generate synthetic dataset for demonstration
X = df.drop(columns=['loan_status'])
y = df['loan_status']

# Split df into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, stratify=y, random_state=42)
eval_set = [(X_test, y_test)]
# Define the experiment name
experiment_name = "XGBoost_Hyperparameter_Tuning"

mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Start MLflow experiment
mlflow.set_experiment(experiment_name)

clf = xgb.sklearn.XGBClassifier(
    objective="binary:logistic",
    learning_rate=0.05,
    seed=9616,
    max_depth=20,
    gamma=10,
    n_estimators=500,
    early_stopping_rounds=20,
    eval_metric='auc')

# Define parameter grid for BayesSearchCV
param_grid = {
    "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.5],
    "max_depth": [3, 5, 7, 10],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9],
    "gamma": [0, 0.1, 0.5, 1],
    "n_estimators": [100, 200, 300, 400, 500]
}

# Define BayesSearchCV with ROC AUC as scoring metric
bayes_search = BayesSearchCV(clf, search_spaces=param_grid, n_iter=50, scoring='roc_auc', cv=5, verbose=1, n_jobs=-1)

# Perform BayesSearchCV
with mlflow.start_run():
    bayes_search.fit(X_train, y_train, eval_set=eval_set, verbose=True)

    # Log parameters
    for param, value in bayes_search.best_params_.items():
        mlflow.log_param(param, value)

    # Log metrics
    mlflow.log_metric("best_roc_auc", bayes_search.best_score_)

    # Evaluate model on test set
    y_pred_proba = bayes_search.predict_proba(X_test)[:, 1]
    test_roc_auc = roc_auc_score(y_test, y_pred_proba)
    mlflow.log_metric("test_roc_auc", test_roc_auc)

    feature_importance = bayes_search.best_estimator_.feature_importances_
    for i, importance in enumerate(feature_importance):
        mlflow.log_metric(f"feature_{i}_importance", importance)

    # Log dataset
    mlflow.log_artifact('data/cleaned_data.csv', artifact_path='datasets')

    # Log model
    mlflow.xgboost.log_model(bayes_search.best_estimator_, "xgboost_model")

    # Prepare ROC AUC Curve
    all_roc_auc_scores = bayes_search.cv_results_['mean_test_score']

    # Sort ROC AUC scores in descending order
    sorted_indices = sorted(range(len(all_roc_auc_scores)), key=lambda i: all_roc_auc_scores[i], reverse=True)

    # Select top 3 ROC AUC scores and their corresponding parameters
    top_3_roc_auc = all_roc_auc_scores[sorted_indices[:3]]
    top_3_params = [bayes_search.cv_results_[f"params"][i] for i in sorted_indices[:3]]

    # Plot ROC curves for the top 3 models
    plt.figure(figsize=(10, 6))
    for i, idx in enumerate(sorted_indices[:3]):
        model = bayes_search.cv_results_['estimator'][idx]
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label=f"Model {i + 1} (AUC = {top_3_roc_auc[i]:.2f})")

    plt.plot([0, 1], [0, 1], linestyle='--', color='grey', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Top 3 Models')
    plt.legend()

    # Save plot as image
    plot_path = "top_3_roc_curves_plot.png"
    plt.savefig(plot_path)

    # Log plot as artifact in MLflow
    mlflow.log_artifact(plot_path, "plots")

    mlflow.end_run()