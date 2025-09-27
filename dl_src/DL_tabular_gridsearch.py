import json
import argparse
import pandas as pd
import os
import sys
from datetime import datetime
sys.path.append('../ml_src/src')

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import BernoulliRBM

from models import CNNCustom, RNNCustom, LSTMCustom, GRUCustom, GCNCustom
from metrics import get_performance


# Define grid search
def get_search_params(model_name, target_col):
    if model_name == 'CNN':
        return {
            'n_features': [27],
            'kernel_size': [3],
            'n_layers': [1 , 2, 3],
            'n_epochs': [10, 20, 50],
            'lr': [1e-2, 1e-3, 1e-4],
            'n_classes': [4 if target_col == 'Severity' else 2]
        }
    
    if model_name == 'LSTM' or model_name == 'GRU' or model_name == 'RNN': 
        return {
            'input_dim': [1],
            'hidden_dim': [8, 16],
            'n_layers': [1, 2],
            'n_epochs': [20, 50],
            'lr': [1e-2, 1e-4],
            'n_classes': [4 if target_col == 'Severity' else 2]
        }
    
    if model_name == 'GCN':
        return {
            'n_features': [27],
            'in_features': [1],
            'hidden_dim': [8, 16],
            'n_layers': [1, 2, 3],
            'n_classes': [4 if target_col == 'Severity' else 2],
            'n_epochs': [20, 50],
            'lr': [1e-2, 1e-4]
        }

def get_model(model_name):
    if model_name == 'CNN':
        return CNNCustom(n_features=27, kernel_size=3, n_layers=2, n_classes=2, n_epochs=10)
    
    if model_name == 'RNN':
        return RNNCustom(input_dim=1, hidden_dim=8, n_layers=1, n_classes=2, n_epochs=10)
    
    if model_name == 'LSTM':
        return LSTMCustom(input_dim=1, hidden_dim=8, n_layers=1, n_classes=2, n_epochs=10)
    
    if model_name == 'GRU':
        return GRUCustom(input_dim=1, hidden_dim=8, n_layers=1, n_classes=2, n_epochs=10)

    if model_name == 'GCN':
        return GCNCustom(n_features=27, in_features=1, hidden_dim=8, n_layers=1, n_classes=2, n_epochs=10)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help="Input name of a model")
    parser.add_argument('--target_col', type=str, help="Choose thresshold for process", choices=["Severity", "AHI_5", "AHI_15", "AHI_30"])

    return parser


def get_dataset(target_col, data_path="../data/SHHS_1_complete_patients.csv"):
    df = pd.read_csv(data_path)
    df.head(5)

    # I am going to add columns AHI5, AHI15, and AHI30 
    df['AHI_5'] = df['Severity'].apply(lambda x: 1 if x >= 1 else 0)
    df['AHI_15'] = df['Severity'].apply(lambda x: 1 if x >= 2 else 0)
    df['AHI_30'] = df['Severity'].apply(lambda x: 1 if x >= 3 else 0)

    return df.iloc[:, 0:27].values, df[target_col].values


def tuning(model_name, target_col):

    # Initialize paths
    if not os.path.isdir(os.path.join('../results', target_col)):
        os.mkdir(os.path.join('../results', target_col))

    if not os.path.isdir(os.path.join('../results', target_col, model_name)):
        os.mkdir(os.path.join('../results', target_col, model_name))

    result_path = os.path.join("../results", target_col, model_name, model_name + "_metrics.csv")
    best_param_path = os.path.join("../results", target_col, model_name, model_name + "_best_params.json")

    # data processing
    x, y = get_dataset(target_col=target_col)

    best_params = dict()
    performance = None

    # Hyperparameter tuning
    kf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
    for fold, (train, test) in enumerate(kf.split(x,y)):
        print(f"**********Hyperparameter tuning for fold {fold}***************************")
        print('Params: ', get_search_params(model_name, target_col))
        scaler = MinMaxScaler()
        scaler.fit(x[train])

        grid_lr = GridSearchCV(estimator=get_model(model_name), param_grid=get_search_params(model_name, target_col), scoring='f1_macro', cv=5, n_jobs=5, verbose=4)
        grid_lr.fit(scaler.transform(x[train]), y[train])

        best_param = grid_lr.best_params_
        train_predict = grid_lr.best_estimator_.predict(scaler.transform(x[train]))
        test_predict = grid_lr.best_estimator_.predict(scaler.transform(x[test]))

        train_performance = get_performance(model_name, fold, is_train=True, y_pred=train_predict, y_true=y[train])
        test_performance = get_performance(model_name, fold, is_train=False, y_pred=test_predict, y_true=y[test])

        # Save performance
        best_params["fold_" + str(fold+1)] = best_param
        df1 = pd.DataFrame(data=train_performance, index=[0])
        df2 = pd.DataFrame(data=test_performance, index=[0])
        if performance is None:
            performance = pd.concat([df1, df2])
        else:
            performance = pd.concat([performance, df1])
            performance = pd.concat([performance, df2])


        # Save in files
        with open(best_param_path, "w") as f:
            json.dump(best_params, f, indent=2)

        performance.sort_values(by=["is_train", "fold"], inplace=True)
        performance.reset_index(drop=True)
        performance.to_csv(result_path, index=False)
        print(f"Complete hyperparameter tuning for {model_name} !")


if __name__ == '__main__':
    start_time = datetime.now()
    parser = parse_arguments()
    args = parser.parse_args()
    tuning(args.model_name, args.target_col)
    end_time = datetime.now()
    print(f"Grid search test for {args.model_name} completes in {end_time - start_time}")

    with open("../logs/log.txt", 'a') as f:
        f.write(f"Grid search test for {args.model_name} completes in {end_time - start_time}\n")
