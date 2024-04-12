"""
Model_Demo.py file to demonstrate the prediction of the best trained model
"""
import time
from typing import Any
import seaborn as sns
import mlflow
import pandas as pd
from pandas import DataFrame
from sklearn.metrics import accuracy_score, f1_score, roc_curve, classification_report, precision_recall_curve, \
    roc_auc_score
from utils.log import LOGGER
import matplotlib.pyplot as plt
import xgboost as xgb



def get_processed_data() -> tuple[DataFrame, DataFrame, Any]:
    """
    Load the data and split it into features and target column
    :return: Dataframe, features and target column
    """
    LOGGER.info('Reading data...')
    df = pd.read_csv('../data/demo_data.csv', low_memory=False)
    y = df['loan_status']
    X = df.drop('loan_status', axis=1)
    return df, X, y


def get_model_and_predict(X: pd.DataFrame, modelpath) -> tuple[int | Any, Any, Any]:
    """
    Load the best trained model from mlflow and predict the probability of default
    :param X: Features DataFrame
    :return: probabilities, predictions and the model
    """
    LOGGER.info('Model is predicting...')
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    logged_model = modelpath
    loaded_model = mlflow.xgboost.load_model(logged_model)
    probas = loaded_model.predict_proba(pd.DataFrame(X))
    preds = loaded_model.predict(pd.DataFrame(X))
    probas = probas * 100
    return probas, preds, loaded_model


def add_save_predictions(probas, df: pd.DataFrame):
    """
    Add probablility of default to the source dataframe
    :param probas: predicted probabilities of the model
    :param df: source Dataframe
    :return:
    """
    df["prediction"] = probas[:, 1]
    df["prediction"] = round(df["prediction"], 2)
    savepath = 'data/predicted_data.csv'
    df.to_csv('../data/predicted_data.csv', index=False)
    LOGGER.info(f'Dataset with a Predictions column as Probability of Default saved in {savepath}')


def calculate_metrics(probas, preds, y):
    """
    Calculate the accuracy, F1 score and ROC_AUC as well as classification report
    :param probas: predicted probabilities of the model
    :param preds: binary predictions of the model
    :param y: target column
    :return:
    """
    LOGGER.info('Calculating metrics...')
    probas_positive = probas[:, 1]

    # Calculate the best ROC AUC score using the best threshold
    roc_auc_best = roc_auc_score(y, probas_positive)
    accuracy = accuracy_score(y, preds)
    f1 = f1_score(y, preds)

    print(f'The trained XGBoost model achieved an accuracy of {accuracy:.2f}')
    print(f'The trained XGBoost model achieved an F1 score of {f1:.2f}')
    print(f'The trained XGBoost model achieved an ROC AUC score of {roc_auc_best:.4f}')
    time.sleep(6)
    print('\nClassification Report:')
    print(classification_report(y, preds))


def plot_roc_curve(y_test, probas):
    """
    Plot the roc auc curve of the model
    :param y_test: test target column
    :param probas: predicted probabilities of the model
    :return:
    """
    fpr, tpr, _ = roc_curve(y_test, probas)
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()


def plot_precision_recall_curve(y_test, probas):
    """
    Plot the precision recall curve of the model
    :param y_test: test target column
    :param probas: predicted probabilities of the model
    :return:
    """
    precision, recall, _ = precision_recall_curve(y_test, probas)
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.show()



def plot_probability_distribution(probas):
    """
    Plot the probability distribution
    :param probas: predicted probabilities of the model
    :return:
    """
    plt.hist(probas[:, 1], bins=10, color='blue', alpha=0.7)
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.title('Probability Distribution Histogram')
    plt.show()


def feature_importance(model):
    """
    Calculate the feature importances
    :param model: used xgboost model
    :return:
    """
    xgb.plot_importance(model)
    plt.show()


def predict():
    """
    Predict loan default probability using xgboost model
    :return:
    """
    LOGGER.info('Model demo starting...')
    time.sleep(5)
    df, X, y = get_processed_data()
    probas, preds, loaded_model = get_model_and_predict(X, 'runs:/97f5ce7792264e00aa54ee1e738b817c/xgboost_model')
    add_save_predictions(probas, df)
    calculate_metrics(probas, preds, y)
    colors = ['#0476df', '#50b1ff', '#0458a5', '#88cbff', '#00457a', '#032a4d', '#9e9e9e', '#828282', '#0078d6']
    sns.set_palette(sns.color_palette(colors))
    plot_roc_curve(y, probas[:, 1])
    plot_precision_recall_curve(y, probas[:, 1])
    plot_probability_distribution(probas)
    feature_importance(loaded_model)
    LOGGER.info('Model Demo Completed.')


if __name__ == '__main__':
    predict()
