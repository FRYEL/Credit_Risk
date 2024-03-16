import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import roc_auc_score
import mlflow
import mlflow.xgboost

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

clf.fit(X_train, y_train, eval_set=eval_set, verbose=True)

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
random_search = RandomizedSearchCV(clf, param_distributions=param_grid, n_iter=150, scoring='roc_auc', cv=5, verbose=1,
                                   n_jobs=-1)

# Perform RandomizedSearchCV
with mlflow.start_run():
    random_search.fit(X_train, y_train, eval_set=eval_set, verbose=True)

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
