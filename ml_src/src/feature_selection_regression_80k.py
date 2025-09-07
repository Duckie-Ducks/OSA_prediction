import argparse
import pandas as pd
import os
from datetime import datetime
import json
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from imblearn.over_sampling import ADASYN, SMOTE
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler

from metrics import get_performance, get_regression_performance
from models import get_model
from utils import MatrixExcelSaver, create_folder, DictExcelSaver


def get_dataset(target_col, data_path="/home/ndoan01/OSA_ML/data/OSA_classifier_AHI.xlsx"):
    df = pd.read_excel(data_path)

    df = df.iloc[:, 1:]

    return df.iloc[:, 0:49].values, df.iloc[:, 49].values, df.iloc[:, 50].values


def get_param(model_name, fold, target_col):
    data_path = os.path.join("../results", target_col, model_name, model_name + '_best_params.json')
    with open(data_path, 'r') as f:
        params = json.load(f)

    return params["fold_" + str(fold)]


def get_feature(feature_name):
    with open("../feature_config/feature_config.json") as f:
        feature_collections = json.load(f)

    return feature_collections[feature_name]


def get_balance_strategy(balance_strategy):
    if balance_strategy is None:
        return None
    elif balance_strategy == "adasyn":
        return ADASYN(random_state=0)
    elif balance_strategy == "smote":
        return SMOTE(random_state=0)
    elif balance_strategy == "borderline":
        return BorderlineSMOTE(random_state=0)
    elif balance_strategy == "underline":
        return RandomUnderSampler(random_state=0)

def feature_selection(model_name, target_col, balance_strategy, features_name):
    result_path = os.path.join('../results', target_col, model_name, '80k_' + model_name+'_'+balance_strategy+'_'+features_name+'.csv')
    pred_path = os.path.join('../results', target_col, model_name, '80k_'+model_name+'_'+balance_strategy+'_'+features_name+'_pred.xlsx')
    features = get_feature(feature_name=features_name)
    preds = dict()
    x, y, z = get_dataset(target_col=target_col)

    performance = None

    # Hyperparameter tuning
    kf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
    for fold, (train, test) in enumerate(kf.split(x,y)):
        # Normalize data
        scaler = MinMaxScaler()
        scaler.fit(x[train])
        x_scale = scaler.transform(x)
        label_scaler = StandardScaler()
        label_scaler.fit(z[train].reshape(-1, 1))

        # Model initialize weights
        model = get_model(model_name)
        param = get_param(model_name, fold+1, target_col)

        model.set_params(**param)

        # Balancing
        balancer = get_balance_strategy(balance_strategy)

        if balancer is not None:
            X_train_balanced, y_train_balanced = balancer.fit_resample(x_scale[train], label_scaler.transform(z[train].reshape(-1, 1))[:, 0])
            model.fit(X_train_balanced, y_train_balanced)
        else:
            model.fit(x_scale[train], label_scaler.transform(z[train].reshape(-1, 1))[:, 0])

        train_predict = model.predict(x_scale[train])
        train_predict = label_scaler.inverse_transform(train_predict.reshape(-1, 1))[:, 0]
        test_predict = model.predict(x_scale[test])
        test_predict = label_scaler.inverse_transform(test_predict.reshape(-1, 1))[:, 0]

        preds[f'fold_{fold+1}_pred'] = test_predict.tolist()
        preds[f'fold_{fold+1}_label'] = z[test].tolist()

        train_performance = get_regression_performance(model_name, fold, is_train=True, y_pred=train_predict, y_true=z[train])
        test_performance = get_regression_performance(model_name, fold, is_train=False, y_pred=test_predict, y_true=z[test])

        
        create_folder(os.path.join('../results', target_col, model_name))
        DictExcelSaver.save(preds, pred_path)

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
    print("------------------------------------------------------------------")
    print(f"Complete feature selection for {model_name} - {balance_strategy} - {features_name} - {balance_strategy} !")

    pass

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help="Input name of a model")
    parser.add_argument('--target_col', type=str, help="Choose thresshold for process", choices=["Regression"])
    parser.add_argument('--balance_strategy', type=str, help="Choose strategy for balancing dataset", choices=["adasyn", "smote", "None", "borderline", "underline"])
    parser.add_argument('--features_name', type=str, help="Choose collection of features for learning", choices=["all_features", "Demographic", "Measurements", "Symptoms", "Questionnaires", 'Comorbidities', 'Mencar', 'Huang', 'Rodruiges', 'KBest Chi2', 'RF Impurity', 'RF Permutation', 'KBest Fclass', 'Correlation', 'Kruskall Chi', 'SHAP_RF', 'CatBoost', 'Wu', 'Ustun'])

    return parser


if __name__ == '__main__':
    start_time = datetime.now()
    parser = parse_arguments()
    args = parser.parse_args()
    feature_selection(args.model_name, args.target_col, args.balance_strategy, args.features_name)
    end_time = datetime.now()
    print(f"Grid search test for {args.model_name}-{args.target_col}-{args.balance_strategy}-{args.features_name} completes in {end_time - start_time}")

    with open("../logs/log_balancing.txt", 'a') as f:
        f.write(f"Feature selection for {args.model_name} completes in {end_time - start_time}\n")
    pass
