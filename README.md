# Probability of Loan Default Prediction with an XGBoost Model

## Overview

This repository contains Python scripts for training an XGBoost model to predict loan default probabilities and
demonstrating its predictions. The model is trained using Bayesian optimization (`BayesSearchCV`) and the dataset is
preprocessed using the `preprocessing.py` script. The best model parameters and predictions are logged using MLflow.

- `Model_training.py`: This script trains the XGBoost model, preprocesses the dataset, and logs the best model
  parameters and predictions using MLflow.
- `Model_Demo.py`: This script demonstrates how to load the trained model using MLflow and use it for predictions. It
  calculates various performance metrics and visualizes the results.
- `preprocessing.py`: This script preprocesses the raw source data by cleaning the dataset, handling missing values,
  encoding categorical features, and creating new features.
- `Notebooks`: Includes Notebooks which have been used to test preprocessing steps, run model testing, encoding
  comparisons and visualizations.

## Dependencies

The scripts rely on the following Python libraries, specified in `requirements.txt`:

- xgboost~=2.0.3
- shap~=0.42.1
- mlflow~=2.9.2
- scikit-learn~=1.4.1.post1
- pandas~=2.1.4
- matplotlib~=3.8.0
- numpy~=1.26.4
- kneed~=0.8.5
- category_encoders~=2.6.0
- seaborn~=0.12.2
- kaggle~=1.6.6
- scikit-optimize~=0.10.1
- python-dotenv~=0.21.0

A python version of 3.8 or higher is recommended.

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/FRYEL/Credit_Risk
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your Kaggle API credentials by creating a `.env` file in the project root directory and adding your Kaggle
   username and key:
   ```
   KAGGLE_USERNAME=your_kaggle_username
   KAGGLE_KEY=your_kaggle_api_key
   ```

4. Ensure that the MLflow tracking server is running locally on `http://127.0.0.1:5000`. You can start the MLflow server
   using:
   ```bash
   mlflow ui
   ```
5. Note: if the mlflow server is already running or doesnt stop when unsing control+c, run this code:
   ```bash
   pkill -f gunicorn
   ```

## Usage

### 1. Data Preprocessing and Model Training (`Model_training.py`)

This script will download the dataset from Kaggle, preprocess it using the`preprocessing.py`file and
generate the `cleaned_data.csv`, split it into training and validation sets, and then train the XGBoost model using
Bayesian optimization. The best model parameters are logged using MLflow.

### 2. Model Demo (`Model_Demo.py`)

This script will load the best trained model using MLflow, make predictions on the processed data, calculate various
metrics, and visualize the results.

### 3. Combined (`main.py`)

To run both scripts you can execute the `main.py` file.

## Parameters

### Model_training

- `set_param_space()`: You can individually set up the parameter space for the BayesSearchCV
- `run_experiment()`: This function includes a boolean to reduce the intial dataset to a smaller size. The percentage
  can be set inside the `runtime_split()` function call. Here you can also set the train size which automatically sets
  the validation size to 0.5 of the train. Then you can set the iterations and cross-validations (Default 100 iters and
  5 CV)

## File Structure

```
.
├── data
│   ├── cleaned_data.csv (local)
│   ├── predicted_data.csv (local)
│   ├── demo_data.csv (local)
│   └── loan/ (local)
│        └── loan.csv (local)
├── mlartifacts/
├── mlruns/
├── Notebooks/
│   ├── Data_preprocessing.ipynb
│   ├── dummy_classifier.ipynb
│   ├── Model_testing.ipynb
│   ├── Runtime_test.ipynb
│   └── Viz.ipynb
├── src/
│   ├── main.py
│   ├── Model_Demo.py
│   ├── Model_training.py
│   └── win64_model/
│       └── model_training_win64.py
├── utils/
│    ├── log.py
│    ├── preprocessing.py
│    └── visualizations.py
├── .env (local)
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

- `data`: Contains the cleaned dataset (`cleaned_data.csv`), the predicted dataset (`predicted_data.csv`) and the source
  data (`loan.csv`).
- `utils`: Contains the preprocessing script (`preprocessing.py`) and other utilities.
-
    - `preprocessing.py`: Script for preprocessing the raw source data.
- `src`:
    - `Model_training.py`: Script for training the XGBoost model.
    - `Model_Demo.py`: Script for demonstrating the prediction using the trained model.
    - The `win64_model` directory is used for the higher performance desktop hardware and doesn't distinguish itself
      with the exception of a different mlflow project name
- `requirements.txt`: List of required Python packages.
- `README.md`: This README file.

## Authors

- [Furkan Yel]


