"""
main.py file to train a xgboost model, to predict loan default probabilities.
:dependencies: preprocessing.py in utils
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
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import numpy as np
from sklearn.model_selection import train_test_split


def get_data():
    """
    Load the source data and unzip
    :return: unprocessed data
    """
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
    """
    Load the data and preprocess it
    :return: preprocessed data
    """
    data = get_data()
    out = preprocess_data(data)
    return out


def runtime_split(data, df_size):
    """
    Reduce the size of the initial dataset for runtime improvements
    :param data: unprocessed dataset
    :param df_size: desired size of the reduced dataset
    :return:
    """
    # Splitting the DataFrame
    X_train, X_test, y_train, y_test = train_test_split(data.drop('loan_status', axis=1),
                                                        data['loan_status'],
                                                        test_size=df_size,
                                                        stratify=data['loan_status'],
                                                        random_state=42)

    # Merging back into one DataFrame
    reduced_df = pd.concat([X_test, y_test], axis=1)
    return reduced_df


def prepare_split(data, test_size=0.6):
    """
    Split the dataset into train, val and test sets
    :param data: preprocessed dataframe
    :param test_size: choose test size (default=0.6)
    :return: X_train, X_test, y_train, y_test and eval set (X_val, y_val)
    """

    X = data.drop('loan_status', axis=1)
    y = data['loan_status']
    # Instantiate StratifiedShuffleSplit with n_splits=1
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    LOGGER.info('Splitting data with StratifiedShuffleSplit...')
    # Split data into train and test
    train_index, test_index = next(sss.split(X, y))
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    LOGGER.info('Getting test, train and validation sets...')
    # Further split test data into validation and test
    sss_val_test = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_index, test_index = next(sss_val_test.split(X_test, y_test))
    X_val, X_test = X_test.iloc[val_index], X_test.iloc[test_index]
    y_val, y_test = y_test.iloc[val_index], y_test.iloc[test_index]

    eval_set = [(X_val, y_val)]
    return X_train, X_test, y_train, y_test, eval_set, test_size


def set_mlflow_uri():
    experiment_name = "XGBoost_Bayes_HT"
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(experiment_name)
    LOGGER.info(f'Setting mlflow uri for {experiment_name}...')


def set_model():
    """
    Sets up the xgboost model classifier
    :return: xgboost classifier
    """
    clf = xgb.sklearn.XGBClassifier(
        objective="binary:logistic",
        seed=7777,
        eval_metric='auc',
        early_stopping_rounds=20)
    return clf


def set_param_space():
    """
    Sets up the parameter space for the bayessearchCV
    :return: parameter space
    """
    param_space = {
        "learning_rate": [0.01, 0.03, 0.04, 0.05, 0.1, 0.25, 0.35, 0.5],
        "max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9],
        "gamma": [0, 0.25, 0.5, 0.75, 1],
        "n_estimators": [100, 200, 300, 400, 450, 500]
    }
    return param_space


def model_tuning(X_train, y_train, eval_set, iterations=100, cv=5):
    """
    Trains the xgboost model with bayessearchCV
    :param cv: value for cross validation, defaults to 5
    :param iterations: value of iterations for bayes search, defaults to 100
    :param X_train: trainings data
    :param y_train: trainings target column
    :param eval_set: X_val and y_val sets
    :return: fitted BayesSearchCV
    """
    clf = set_model()
    param_space = set_param_space()

    mlflow.start_run()
    LOGGER.info('initiating BayesSearchCV...')
    bayes_search = BayesSearchCV(clf, search_spaces=param_space,
                                 n_iter=iterations, scoring='roc_auc',
                                 cv=cv, verbose=1,
                                 n_jobs=-1)

    bayes_search.fit(X_train, y_train, eval_set=eval_set, verbose=True)

    mlflow.xgboost.log_model(bayes_search.best_estimator_, "xgboost_model")

    return bayes_search


def create_plots(bayes_search, X_train, y_train, X_test, y_test):
    """
    Creates the ROC AUC curve plot
    :param bayes_search: fitted BayesSearchCV
    :param X_train: trainings data
    :param y_train: trainings target data
    :param X_test: test data
    :param y_test: test target data
    :return: shows ROC AUC curve plot
    """
    all_roc_auc_scores = bayes_search.cv_results_['mean_test_score']

    # Find the index of the top-performing model
    best_index = np.argmax(all_roc_auc_scores)
    best_params = bayes_search.cv_results_['params'][best_index]
    best_roc_auc = all_roc_auc_scores[best_index]

    # Plot ROC curve for the best model
    plt.figure(figsize=(10, 6))
    model = xgb.sklearn.XGBClassifier(**best_params)
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f"Best Model (AUC = {best_roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], linestyle='--', color='grey', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Best Model')
    plt.legend()
    plt.show()
    plot_path = "best_roc_curve_plot.png"
    plt.savefig(plot_path)

    # Log plot as artifact in MLflow
    mlflow.log_artifact(plot_path, "plots")


def mlflow_logging(bayes_search, X_test, y_test, test_size):
    """
    Logs the training params, best_train auc , test auc, feature importance, dataset
    and xgboost model
    :param test_size: test size of the split
    :param bayes_search: fitted BayesSearchCV
    :param X_test: test data
    :param y_test: test target data
    :return: nothing
    """
    # Log parameters
    for param, value in bayes_search.best_params_.items():
        mlflow.log_param(param, value)

    split_ratio = f"train_size={1 - test_size}, test_size={test_size / 2}, val_size={test_size / 2}"
    mlflow.log_param("split_ratio", split_ratio)

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
    mlflow.log_artifact("./data/cleaned_data.csv")
    LOGGER.info('Run Completed...')


def run_experiment():
    """
    Run the experiment to train a xgboost model for loan default prediction
    :return:
    """
    processed_data = data_pipeline()
    # SET TRUE OR FALSE TO USE runtime_split TO REDUCE INITIAL DATASET
    reduce_data = True
    if reduce_data:
        LOGGER.info('Processed data is split for performance...')

        reduced_dataset = runtime_split(processed_data, 0.1)
        X_train, X_test, y_train, y_test, eval_set, test_size = prepare_split(reduced_dataset, 0.6)
    else:
        X_train, X_test, y_train, y_test, eval_set, test_size = prepare_split(processed_data, 0.6)
    set_mlflow_uri()
    model = model_tuning(X_train, y_train, eval_set, iterations=100, cv=5)
    mlflow_logging(model, X_test, y_test, test_size)
    create_plots(model, X_train, y_train, X_test, y_test)


if __name__ == '__main__':
    run_experiment()
