import xgboost as xgb
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import roc_auc_score
import mlflow
import mlflow.xgboost
from utils.log import *

df = pd.read_csv('../df/cleaned_data.csv')

# Generate synthetic dataset for demonstration
X, y = df.drop(columns=['default']), df['default']

# Split df into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the experiment name
experiment_name = "XGBoost_Hyperparameter_Tuning"

# Start MLflow experiment
mlflow.set_experiment(experiment_name)

# Define the classifier
clf = xgb.XGBClassifier(objective="binary:logistic", seed=42)

# Define parameter grid for RandomizedSearchCV
param_grid = {
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 5, 7],
    "subsample": [0.5, 0.7, 0.9],
    "colsample_bytree": [0.5, 0.7, 0.9],
    "gamma": [0, 1, 5],
    "n_estimators": [100, 200, 300]
}

# Define RandomizedSearchCV with ROC AUC as scoring metric
random_search = RandomizedSearchCV(clf, param_distributions=param_grid, n_iter=50, scoring='roc_auc', cv=5, verbose=1,
                                   n_jobs=-1)

# Perform RandomizedSearchCV
with mlflow.start_run():
    random_search.fit(X_train, y_train)

    # Log parameters
    for param, value in random_search.best_params_.items():
        mlflow.log_param(param, value)

    # Log metrics
    mlflow.log_metric("best_roc_auc", random_search.best_score_)

    # Evaluate model on test set
    y_pred_proba = random_search.predict_proba(X_test)[:, 1]
    test_roc_auc = roc_auc_score(y_test, y_pred_proba)
    mlflow.log_metric("test_roc_auc", test_roc_auc)

    # Log model
    mlflow.xgboost.log_model(random_search.best_estimator_, "xgboost_model")
