import numpy as np
from utils.log import LOGGER
import category_encoders as ce

def preprocess_data(df):

    # Filter for individual applications
    df_indv = df[df["application_type"] != "JOINT"]

    # Extract relevant columns to new df
    df_indv.drop(columns=df_indv.columns.difference(
        ['id', 'member_id', 'loan_amnt', 'funded_amnt', 'term', 'int_rate', 'installment',
         'grade', 'sub_grade', 'emp_length', 'home_ownership', 'annual_inc', 'verification_status', 'loan_status',
         'purpose', 'addr_state', 'dti', 'delinq_2yrs', 'mths_since_last_delinq', 'total_acc', 'out_prncp',
         'total_pymnt', 'total_rec_prncp', 'total_rec_interest', 'total_rec_late_fee', 'acc_now_delinq']), inplace=True)

    # Fill any numeric columns na with 0
    for column in df_indv.columns:
        if df_indv[column].isnull().any() and df_indv[column].dtype in ['int64', 'float64']:
            df_indv[column].fillna(0, inplace=True)
    df_indv.isnull().sum()

    LOGGER.info("Preprocessing data...")

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

    df_indv['emp_length'] = df_indv['emp_length'].map(emp_length_map)

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
        'A1': 1,
        'A2': 2,
        'A3': 3,
        'A4': 4,
        'A5': 5,
        'B1': 6,
        'B2': 7,
        'B3': 8,
        'B4': 9,
        'B5': 10,
        'C1': 11,
        'C2': 12,
        'C3': 13,
        'C4': 14,
        'C5': 15,
        'D1': 16,
        'D2': 17,
        'D3': 18,
        'D4': 19,
        'D5': 20,
        'E1': 21,
        'E2': 22,
        'E3': 23,
        'E4': 24,
        'E5': 25,
        'F1': 26,
        'F2': 27,
        'F3': 28,
        'F4': 29,
        'F5': 30,
        'G1': 31,
        'G2': 32,
        'G3': 33,
        'G4': 34,
        'G5': 35
    }

    df_indv['grade'] = df_indv['grade'].map(grades)
    df_indv['sub_grade'] = df_indv['sub_grade'].map(subgrades)
    df_indv['term'] = df_indv['term'].str.extract('(\d+)').astype(int)

    homeownership = {
        'ANY': 0,
        'MORTGAGE': -1,
        'NONE': 0,
        'OTHER': 0,
        'OWN': 2,
        'RENT': 1
    }

    df_indv['home_ownership'] = df_indv['home_ownership'].map(homeownership)

    verification = {
        'Not Verified': -1,
        'Source Verified': 1,
        'Verified': 2
    }

    df_indv['verification_status'] = df_indv['verification_status'].map(verification)

    l_stat = {
        'Charged Off': 1,
        'Default': 1,
        'Does not meet the credit policy. Status:Charged Off': 0,
        'Late (16-30 days)': 1,
        'Late (31-120 days)': 1,
        'Current': 0,
        'Does not meet the credit policy. Status:Fully Paid': 0,
        'Fully Paid': 0,
        'In Grace Period': 0,
        'Issued': 0,
    }

    df_indv['loan_status'] = df_indv['loan_status'].map(l_stat)

    LOGGER.info('Encoding categorical features...')

    encoder1 = ce.CountEncoder(cols=['purpose'])

    # Fit and transform the data
    df_indv = encoder1.fit_transform(df_indv)

    encoder2 = ce.TargetEncoder(cols=['addr_state'])

    # Fit and transform the data
    df_indv = encoder2.fit_transform(df_indv, df_indv['loan_status'])

    LOGGER.info('Engineering features...')

    df_indv['loan_to_income'] = round(df_indv['funded_amnt'] / df_indv['annual_inc'], 2)
    df_indv['loan_to_income'].replace(np.inf, 2, inplace=True)

    df_indv['total_interest'] = round((df_indv['term'] / 12) * df_indv['loan_amnt'] * (df_indv['int_rate'] / 100), 2)

    df_indv['loan_performance'] = round(df_indv['total_pymnt'] - df_indv['funded_amnt'], 2)

    df_indv['repayment_rate'] = round(df_indv['total_pymnt'] / df_indv['funded_amnt'], 2)

    columns = ['id', 'member_id', 'loan_amnt', 'funded_amnt', 'term', 'int_rate',
               'installment', 'grade', 'sub_grade', 'emp_length', 'home_ownership',
               'annual_inc', 'verification_status', 'purpose', 'addr_state', 'dti',
               'delinq_2yrs', 'mths_since_last_delinq',
               'total_acc', 'out_prncp', 'total_pymnt', 'total_rec_prncp',
               'total_rec_late_fee', 'acc_now_delinq', 'loan_to_income',
               'total_interest', 'loan_performance', 'repayment_rate', 'loan_status']

    df_indv = df_indv[columns]

    LOGGER.info('Saving preprocessed data...')
    df_indv.to_csv("./data/cleaned_data.csv", index=False)

    LOGGER.info("Preprocessing finished!")
    return df_indv


