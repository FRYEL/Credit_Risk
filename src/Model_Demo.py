"""
Model_Demo.py file to demonstrate the prediction of the best trained model
"""
import time
import mlflow
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_curve, classification_report, precision_recall_curve, roc_auc_score, confusion_matrix
from utils.log import LOGGER
import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np

def get_processed_data():
    LOGGER.info('Reading data...')
    df = pd.read_csv('../data/cleaned_data.csv', low_memory=False)
    y = df['loan_status']
    X = df.drop('loan_status', axis=1)
    return df, X, y


def get_model_and_predict(X):
    LOGGER.info('Model is predicting...')
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    logged_model = 'runs:/8510bb96364e476da1d5a4960623d6da/xgboost_model'
    loaded_model = mlflow.xgboost.load_model(logged_model)
    probas = loaded_model.predict_proba(pd.DataFrame(X))
    preds = loaded_model.predict(pd.DataFrame(X))
    probas = probas * 100
    return probas, preds, loaded_model


def add_save_predictions(probas, df):
    df["prediction"] = probas[:, 1]
    df["prediction"] = round(df["prediction"], 2)
    savepath = 'data/predicted_data.csv'
    df.to_csv('../data/predicted_data.csv', index=False)
    LOGGER.info(f'Dataset with a Predictions column as Probability of Default saved in {savepath}')


def calculate_metrics(probas, preds, y, loaded_model):
    LOGGER.info('Calculating metrics...')
    probas_positive = probas[:, 1]
    # Calculate fpr, tpr and thresholds for the ROC curve
    fpr, tpr, thresholds = roc_curve(y, probas_positive)
    best_threshold_index = np.argmax(tpr - fpr)

    # Calculate the best ROC AUC score using the best threshold
    roc_auc_best = roc_auc_score(y, probas_positive)
    accuracy = accuracy_score(y, preds)
    f1 = f1_score(y, preds)

    print(f'The trained XGBoost model achieved an accuracy of {accuracy * 100:.2f}%')
    print(f'The trained XGBoost model achieved an F1 score of {f1*100:.2f}%')
    print(f'The trained XGBoost model achieved an ROC AUC score of {roc_auc_best*100:.2f}%')
    time.sleep(6)
    print('\nClassification Report:')
    print(classification_report(y, preds))


def plot_roc_curve(y_test, y_pred_proba):
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()


def plot_precision_recall_curve(y_test, y_pred_proba):
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.show()


def plot_probability_distribution(probas):
    plt.hist(probas[:, 1], bins=10, color='blue', alpha=0.7)
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.title('Probability Distribution Histogram')
    plt.show()


def feature_importance(model):
    xgb.plot_importance(model)
    plt.show()


def predict():
    LOGGER.info('Model demo starting...')
    time.sleep(5)
    df, X, y = get_processed_data()
    probas, preds, loaded_model = get_model_and_predict(X)
    add_save_predictions(probas, df)
    calculate_metrics(probas, preds, y, loaded_model)
    plot_roc_curve(y, probas[:, 1])
    plot_precision_recall_curve(y, probas[:, 1])
    plot_probability_distribution(probas)
    feature_importance(loaded_model)
    LOGGER.info('Model Demo Completed.')


if __name__ == '__main__':
    predict()
