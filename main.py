"""
Prediction of loan default probabilities with xGBoost and BayesSearchCV for param tuning
"""

import os
import subprocess
from utils.log import LOGGER
from utils.preprocessing import preprocess_data
import zipfile
import xgboost as xgb
import pandas as pd
from sklearn.metrics import roc_auc_score
from skopt import BayesSearchCV
import mlflow.xgboost
from sklearn.model_selection import StratifiedShuffleSplit


def get_data():
    os.environ['KAGGLE_USERNAME'] = 'furkanyel'
    os.environ['KAGGLE_KEY'] = '854a1d21e333a19a0ea49b3eae8ac61b'

    from kaggle.api.kaggle_api_extended import KaggleApi
    # Authenticate with Kaggle MAKE SURE requirements.txt ARE FULLFILLED
    api = KaggleApi()
    # Define the command
    LOGGER.info(f'Downloading the Dataset...')
    command = "kaggle datasets download -d ranadeep/credit-risk-dataset -p ./data"
    subprocess.run(command, shell=True)

    zip_file = "./data/credit-risk-dataset.zip"
    destination_folder = "./data"

    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(destination_folder)

    unprocessed_data = pd.read_csv('./data/loan/loan.csv', low_memory=False)

    return unprocessed_data


def data_pipeline():
    data = get_data()
    out = preprocess_data(data)
    return out


def prepare_split(data, test_size=0.6):
    X = data.drop('loan_status', axis=1)
    y = data['loan_status']
    # Instantiate StratifiedShuffleSplit with n_splits=1
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)

    # Split data into train and test
    train_index, test_index = next(sss.split(X, y))
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Further split test data into validation and test
    sss_val_test = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_index, test_index = next(sss_val_test.split(X_test, y_test))
    X_val, X_test = X_test.iloc[val_index], X_test.iloc[test_index]
    y_val, y_test = y_test.iloc[val_index], y_test.iloc[test_index]

    eval_set = [(X_val, y_val)]
    return X_train, X_test, y_train, y_test, eval_set


def set_model():
    clf = xgb.sklearn.XGBClassifier(
        objective="binary:logistic",
        seed=7777,
        eval_metric='auc',
        early_stopping_rounds=20)
    return clf


def set_param_space():
    param_space = {
        "learning_rate": [0.01, 0.03, 0.04, 0.05, 0.1],
        "max_depth": [5, 7, 8, 9, 10],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9],
        "gamma": [0.5, 0.75, 1],
        "n_estimators": [300, 400, 500]
    }
    return param_space


def model_tuning(X_train, y_train, eval_set):
    clf = set_model()
    param_space = set_param_space()

    bayes_search = BayesSearchCV(clf, search_spaces=param_space,
                                 n_iter=100, scoring='roc_auc',
                                 cv=5, verbose=1,
                                 n_jobs=-1)
    with mlflow.start_run():
        bayes_search.fit(X_train, y_train, eval_set=eval_set, verbose=True)

    return bayes_search


def mlflow_logging(bayes_search, X_test, y_test):
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
    mlflow.log_artifact('../data/cleaned_data.csv', artifact_path='datasets')
    mlflow.xgboost.log_model(bayes_search.best_estimator_, "xgboost_model")


if __name__ == '__main__':
    processed_data = data_pipeline()
    # X_train, X_test, y_train, y_test, eval_set = prepare_split(processed_data, 0.6)
    # clf = set_model()
    # param_space = set_param_space()
    # model = model_tuning(X_train, y_train, eval_set)
    # mlflow_logging(model, X_test, y_test)
