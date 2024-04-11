"""
preprocessing.py Script to preprocess the data
"""
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from utils.log import LOGGER
import category_encoders as ce
import warnings


def ignore_warnings():
    """
    Ignore all warnings and just the SettingWithCopyWarning
    :param: None
    :return: None
    """
    warnings.filterwarnings("ignore")
    warnings.filterwarnings("ignore", message="A value is trying to be set on a copy of a slice from a DataFrame")


def filter_individual_applications(df):
    """
    Filter the individual applications out
    :param df: dataframe
    :return: filtred dataframe
    """
    return df[df["application_type"] != "JOINT"]


def select_relevant_columns(df):
    """
    Select relevant columns
    :param df: dataframe
    :return: relevant columns in dataframe
    """
    relevant_columns = ['loan_amnt', 'funded_amnt', 'term', 'int_rate', 'installment',
         'grade', 'sub_grade', 'emp_length', 'home_ownership', 'annual_inc', 'verification_status', 'loan_status',
         'purpose', 'addr_state', 'dti', 'delinq_2yrs', 'mths_since_last_delinq', 'total_acc', 'out_prncp',
         'total_pymnt', 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'acc_now_delinq']
    return df[relevant_columns]


def fill_na_with_zero(df):
    """
    Fill any numeric columns na with 0
    :param df: dataframe
    :return: dataframe with all null values in numeric columns filled with 0
    """
    for column in df.columns:
        if df[column].isnull().any() and df[column].dtype in ['int64', 'float64']:
            df[column].fillna(0, inplace=True)
    return df


def map_values(df):
    """
    Map values for various columns
    :param df: dataframe
    :return: dataframe with mapped ordinal values
    """
    emp_length_map = {
        '10+ years': 11,
        '< 1 year': 0,
        '1 year': 1,
        '2 years': 2,
        '3 years': 3,
        '4 years': 4,
        '5 years': 5,
        '6 years': 6,
        '7 years': 7,
        '8 years': 8,
        '9 years': 9,
        np.nan: 0
    }

    grades = {
        'A': 1,
        'B': 2,
        'C': 3,
        'D': 4,
        'E': 5,
        'F': 6,
        'G': 7
    }

    subgrades = {
        'A1': 1, 'A2': 2, 'A3': 3, 'A4': 4, 'A5': 5,
        'B1': 6, 'B2': 7, 'B3': 8, 'B4': 9, 'B5': 10,
        'C1': 11, 'C2': 12, 'C3': 13, 'C4': 14, 'C5': 15,
        'D1': 16, 'D2': 17, 'D3': 18, 'D4': 19, 'D5': 20,
        'E1': 21, 'E2': 22, 'E3': 23, 'E4': 24, 'E5': 25,
        'F1': 26, 'F2': 27, 'F3': 28, 'F4': 29, 'F5': 30,
        'G1': 31, 'G2': 32, 'G3': 33, 'G4': 34, 'G5': 35
    }

    homeownership = {
        'ANY': 0, 'MORTGAGE': -1, 'NONE': 0, 'OTHER': 0, 'OWN': 2, 'RENT': 1
    }

    verification = {
        'Not Verified': -1, 'Source Verified': 1, 'Verified': 2
    }

    l_stat = {
        'Charged Off': 1, 'Default': 1, 'Does not meet the credit policy. Status:Charged Off': 0,
        'Late (16-30 days)': 1, 'Late (31-120 days)': 1, 'Current': 0,
        'Does not meet the credit policy. Status:Fully Paid': 0, 'Fully Paid': 0,
        'In Grace Period': 0, 'Issued': 0,
    }

    df['emp_length'] = df['emp_length'].map(emp_length_map)
    df['grade'] = df['grade'].map(grades)
    df['sub_grade'] = df['sub_grade'].map(subgrades)
    df['term'] = df['term'].str.extract('(\d+)').astype(int)
    df['home_ownership'] = df['home_ownership'].map(homeownership)
    df['verification_status'] = df['verification_status'].map(verification)
    df['loan_status'] = df['loan_status'].map(l_stat)

    return df


def encode_categorical_features(df):
    """
    Encode categorical features using TargetEncoder
    :param df: dataframe
    :return: dataframe with numerically encoded features
    """
    encoder1 = ce.TargetEncoder(cols=['purpose'])
    encoder2 = ce.TargetEncoder(cols=['addr_state'])

    df = encoder1.fit_transform(df, df['loan_status'])
    df = encoder2.fit_transform(df, df['loan_status'])

    return df


def feature_engineering(df):
    """
    Perform feature engineering
    :param df: dataframe
    :return: dataframe with new features
    """
    df['loan_to_income'] = round(df['funded_amnt'] / df['annual_inc'], 2)
    df['loan_to_income'].replace(np.inf, 0, inplace=True)
    df['total_interest'] = round((df['term'] / 12) * df['loan_amnt'] * (df['int_rate'] / 100), 2)
    df['loan_performance'] = round(df['total_pymnt'] - df['funded_amnt'], 2)
    df['repayment_rate'] = round(df['total_pymnt'] / df['funded_amnt'], 2)
    df['dti_month'] = round(df['installment'] / (df['annual_inc'] / 12), 3)
    df['dti_month'].replace(np.inf, 0, inplace=True)

    return df


def format_column_order(df):
    """
    Format column order
    :param df: dataframe
    :return: dataframe with new order
    """
    columns = [
        'loan_amnt', 'funded_amnt', 'term', 'int_rate', 'installment', 'grade',
        'sub_grade', 'emp_length', 'home_ownership', 'annual_inc', 'verification_status',
        'purpose', 'addr_state', 'dti', 'delinq_2yrs', 'mths_since_last_delinq',
        'total_acc', 'out_prncp', 'total_pymnt', 'total_rec_prncp','total_rec_int', 'total_rec_late_fee',
        'acc_now_delinq', 'loan_to_income', 'total_interest', 'loan_performance',
        'repayment_rate', 'dti_month', 'loan_status'
    ]

    return df[columns]


def save_preprocessed_data(df, filepath="../data/cleaned_data.csv"):
    """
    Save preprocessed data
    :param df: dataframe
    :param filepath: filepath to save
    :return: None
    """
    X_train, X_test, y_train, y_test = train_test_split(df,
                                                        df['loan_status'],
                                                        test_size=0.2,
                                                        random_state=77)

    # Merging Demo Dataset for later
    demo_df = pd.concat([X_test, y_test], axis=1)
    demo_df.to_csv("../data/demo_data.csv", index=False)

    # Merging back into one DataFrame
    reduced_df = pd.concat([X_train, y_train], axis=1)
    reduced_df.to_csv(filepath, index=False)


def preprocessing_pipeline(df):
    """
    Preprocessing pipeline
    :param df: dataframe
    :return: final preprocessed dataframe
    """
    ignore_warnings()
    df = filter_individual_applications(df)
    df = select_relevant_columns(df)
    df = fill_na_with_zero(df)
    df = map_values(df)
    df = encode_categorical_features(df)
    df = feature_engineering(df)
    df = format_column_order(df)
    save_preprocessed_data(df)
    LOGGER.info("Preprocessing finished!")


    return df
