import argparse
import pandas as pd
import os
from datetime import datetime
import json
from joblib import parallel_backend
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV

from metrics import get_performance
from models import get_model

import warnings
warnings.filterwarnings('ignore')

from joblib import parallel_backend


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help="Input name of a model")
    parser.add_argument('--target_col', type=str, help="Choose thresshold for process", choices=["Severity", "AHI_5", "AHI_15", "AHI_30"])

    return parser

def get_dataset(target_col, data_path="/home/ndoan01/OSA/data/OSA_complete_patients.csv"):
    df = pd.read_csv(data_path, index_col=["PatientID"])
    df.drop(df.columns[[0]], axis=1, inplace=True)
    df.head(5)

    # I am going to add columns AHI5, AHI15, and AHI30 
    df['AHI_5'] = df['Severity'].apply(lambda x: 1 if x >= 1 else 0)
    df['AHI_15'] = df['Severity'].apply(lambda x: 1 if x >= 2 else 0)
    df['AHI_30'] = df['Severity'].apply(lambda x: 1 if x >= 3 else 0)

    return df.iloc[:, 0:49].values, df[target_col].values

def get_params(model_name, target_col, fold):
    json_path = os.path.join("../results", target_col, model_name, model_name + "_best_params.json")

    with open(json_path, 'r') as f:
        params = json.load(f)


    return params[f'fold_{fold+1}']

def eval(model_name, target_col):
    # Initialize paths
    result_folder = os.path.join("../results/", target_col, model_name)
    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)
    result_path = os.path.join("../results", target_col, model_name, model_name + "_all_metrics.csv")

    # data processing
    x, y = get_dataset(target_col=target_col)
    x = MinMaxScaler().fit_transform(x)

    performance = None

    # Hyperparameter tuning
    kf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
    for fold, (train, test) in enumerate(kf.split(x,y)):
        with parallel_backend("threading", n_jobs=-1):

            model = get_model(model_name)
            params = get_params(model_name, target_col, fold)
            model.set_params(**params)

            model.fit(x[train], y[train])

        train_predict = model.predict(x[train])
        test_predict = model.predict(x[test])
        
        train_performance = get_performance(model_name, fold, is_train=True, y_pred=train_predict, y_true=y[train])
        test_performance = get_performance(model_name, fold, is_train=False, y_pred=test_predict, y_true=y[test])
      
        # Save performance
        df1 = pd.DataFrame(data=train_performance, index=[0])
        df2 = pd.DataFrame(data=test_performance, index=[0])
        if performance is None:
            performance = pd.concat([df1, df2])
        else:
            performance = pd.concat([performance, df1])
            performance = pd.concat([performance, df2])


    # Save in files
    performance.sort_values(by=["is_train", "fold"], inplace=True)
    performance.reset_index(drop=True)
    performance.to_csv(result_path, index=False)
    print(f"Complete eval for {model_name} !")


if __name__ == '__main__':
    start_time = datetime.now()
    parser = parse_arguments()
    args = parser.parse_args()
    eval(args.model_name, args.target_col)
    end_time = datetime.now()
    print(f"Evaluate for {args.model_name} completes in {end_time - start_time}")

    with open("../logs/log.txt", 'a') as f:
        f.write(f"Evaluate for {args.model_name} completes in {end_time - start_time}\n")
