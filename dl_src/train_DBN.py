import numpy as np
import pandas as pd
import argparse
import os
import sys
sys.path.append('../ml_src/src')
from metrics import get_performance

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help="Input name of a model")
    parser.add_argument('--target_col', type=str, help="Choose thresshold for process", choices=["Severity", "AHI_5", "AHI_15", "AHI_30"])

    return parser


# Load the dataset
def get_dataset(target_col, data_path="../data/SHHS_1_complete_patients.csv"):
    df = pd.read_csv(data_path)
    df.head(5)

    # I am going to add columns AHI5, AHI15, and AHI30 
    df['AHI_5'] = df['Severity'].apply(lambda x: 1 if x >= 1 else 0)
    df['AHI_15'] = df['Severity'].apply(lambda x: 1 if x >= 2 else 0)
    df['AHI_30'] = df['Severity'].apply(lambda x: 1 if x >= 3 else 0)

    return df.iloc[:, 0:27].values, df[target_col].values


args = parse_arguments().parse_args()
if not os.path.isdir(os.path.join("../results", args.target_col, 'DBN')):
    os.mkdir(os.path.join("../results", args.target_col, 'DBN'))
result_path = os.path.join("../results", args.target_col, 'DBN', 'DBN' + "_metrics.csv")
target_col = args.target_col
x, y = get_dataset(target_col=target_col)
# Split the dataset into training and test sets
kf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
performance = None

for fold, (train, test) in enumerate(kf.split(x,y)):
    scaler = MinMaxScaler()
    scaler.fit(x[train])

    # Initialize the RBM model
    rbm = BernoulliRBM(n_components=256, learning_rate=0.001, n_iter=100, verbose=1)
    # Initialize the logistic regression model
    logistic = LogisticRegression(max_iter=1000)
    # Create a pipeline that first extracts features using the RBM and then classifies with logistic regression
    dbn_pipeline = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])
    # Train the DBN
    dbn_pipeline.fit(scaler.transform(x[train]), y[train])

    # Evaluate the model on the test set
    train_predict = dbn_pipeline.predict(scaler.transform(x[train]))
    test_predict = dbn_pipeline.predict(scaler.transform(x[test]))

    train_performance = get_performance('DBN', fold, is_train=True, y_pred=train_predict, y_true=y[train])
    test_performance = get_performance('DBN', fold, is_train=False, y_pred=test_predict, y_true=y[test])

    # Save performance
    df1 = pd.DataFrame(data=train_performance, index=[0])
    df2 = pd.DataFrame(data=test_performance, index=[0])
    if performance is None:
        performance = pd.concat([df1, df2])
    else:
        performance = pd.concat([performance, df1])
        performance = pd.concat([performance, df2])

    performance.sort_values(by=["is_train", "fold"], inplace=True)
    performance.reset_index(drop=True)
    performance.to_csv(result_path, index=False)
    print(f"Complete hyperparameter tuning for DBN !")
