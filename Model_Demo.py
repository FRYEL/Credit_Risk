import mlflow
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score


def read_source():
    df = pd.read_csv('data/cleaned_data.csv', low_memory=False)
    y = df['loan_status']
    X = df.drop('loan_status', axis=1)
    return df, X, y


def get_model_and_predict(X):
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    logged_model = 'runs:/8510bb96364e476da1d5a4960623d6da/xgboost_model'
    loaded_model = mlflow.xgboost.load_model(logged_model)
    predictions = loaded_model.predict_proba(pd.DataFrame(X))
    predictions = predictions * 100
    return predictions


def add_save_predictions(predictions, df):
    df["prediction"] = predictions[:, 1]
    df["prediction"] = round(df["prediction"], 2)
    df.to_csv('data/predicted_data.csv', index=False)


def calculate_metrics(predictions, df, y):
    predicted_labels = round(df['prediction'])

    accuracy = accuracy_score(y, predicted_labels)

    print("Accuracy:", accuracy)


def predict():
    df, X, y = read_source()
    preds = get_model_and_predict(X)
    add_save_predictions(preds, df)
    #calculate_metrics(preds, df, y)


if __name__ == '__main__':
    predict()
