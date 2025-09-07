import argparse
import pandas as pd
import os
from datetime import datetime
import json
from joblib import parallel_backend
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.impute import SimpleImputer

from metrics import get_performance
from models import get_model


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help="Input name of a model")
    parser.add_argument('--target_col', type=str, help="Choose thresshold for process", choices=["Severity", "AHI_5", "AHI_15", "AHI_30"])

    return parser


def get_dataset(target_col, data_path="/home/ndoan01/OSA_ML/data/OSA_clean.csv"):

    df = pd.read_csv(data_path)
    df['Severity'] = df['Apnea_hypopnea_index']
    df = df.drop(['Apnea_hypopnea_index'], axis=1)

    df['Severity'] = df['Severity'].apply(lambda x: 0 if x < 5 else x)
    df['Severity'] = df['Severity'].apply(lambda x: 1 if 5 <= x < 15 else x)
    df['Severity'] = df['Severity'].apply(lambda x: 2 if 15 <= x < 30 else x)
    df['Severity'] = df['Severity'].apply(lambda x: 3 if 30 <= x else x)    

    df = df[~df['Severity'].isnull()]
    df.reset_index(drop=True, inplace=True)
    # I am going to add columns AHI5, AHI15, and AHI30 
    df['AHI_5'] = df['Severity'].apply(lambda x: 1 if x >= 1 else 0)
    df['AHI_15'] = df['Severity'].apply(lambda x: 1 if x >= 2 else 0)
    df['AHI_30'] = df['Severity'].apply(lambda x: 1 if x >= 3 else 0)

    return df.iloc[:, 0:91], df[target_col]


def get_search_params(model_name):
    json_path = os.path.join("../config", model_name + ".json")

    with open(json_path, 'r') as f:
        params = json.load(f)


    return params


def tuning(model_name, target_col):

    # Initialize paths
    if not os.path.isdir(os.path.join('../results', target_col)):
        os.mkdir(os.path.join('../results', target_col))

    if not os.path.isdir(os.path.join('../results', target_col, model_name)):
        os.mkdir(os.path.join('../results', target_col, model_name))

    result_path = os.path.join("../results", target_col, model_name, 'impute_hypertuning_' + model_name + "_metrics.csv")
    best_param_path = os.path.join("../results", target_col, model_name, 'impute_hypertuning_' + model_name + "_best_params.json")

    # data processing
    x, y = get_dataset(target_col=target_col)

    best_params = dict()
    performance = None

    # Train test split with null value
    # indices = list(range(len(x)))
    # print(x.isnull().any(axis=1))
    # print(sum(x.isnull().any(axis=1)))
    # exit()

    # Train test split with null value    
    features = ['Sex', 'Age', 'Current_smoker', 'Former_smoker',
       'Sedentary', 'Height', 'Weight', 'Cervical_perimeter',
       'Abdominal_perimeter', 'Systolic_BP', 'Diastolic_BP',
       'Maxillofacial_profile', 'BMI', 'High_BP', 'Asthma', 'Rhinitis', 'COPD',
       'Respiratory_fail', 'Myocardial_infarct', 'Coronary_fail',
       'Arrhythmias', 'Stroke', 'Heart_fail', 'Arteriopathy', 'Gastric_reflux',
       'Glaucoma', 'Diabetes', 'Hypercholesterolemia', 'Hypertriglyceridemia',
       'Hypo(er)thyroidism', 'Depression', 'Obesity', 'Dysmorphology',
       'Restless_Leg_Syndrome', 'Snoring', 'Diurnal_somnolence',
       'Driving_drowsiness', 'Morning_fatigue', 'Morning_headache',
       'Memory_problem', 'Nocturnal_perspiration',
       'Shortness_of_breath_on_exertion', 'Nocturia', 'Drowsiness_accident',
       'Near_miss_accident', 'Respiratory_arrest', 'Epworth_scale',
       'Pichots_scale', 'Depression_scale']
    continuous = ['Age','Height','Weight','Cervical_perimeter','Abdominal_perimeter','Systolic_BP','Diastolic_BP', 
              'BMI','Epworth_scale','Pichots_scale','Depression_scale']
    categorical = [i for i in features if i not in continuous]
    x = x[features]

    notnull_indices = x[~x.isnull().any(axis=1)].index.tolist()
    indices = list(range(len(x)))
    
    test_size = 0.2
    n_test = int(test_size * len(x))

    np.random.seed(0)
    test = list(np.random.choice(notnull_indices, n_test))
    train = list(set(indices) - set(test))
    
    # print(np.histogram(y.values))
    # exit()
        
    # print(set(features).issubset(set(x.columns)))

    train_data, train_label = x.iloc[train, :], y.iloc[train]
    test_data, test_label = x.iloc[test, :], y.iloc[test]

    # Imputation
    imp_cont = SimpleImputer(strategy='median', missing_values=np.nan)  # define imputer for continuous variables
    imp_cont = imp_cont.fit(train_data[continuous])                    # fit imputer on columns
    train_data[continuous] = imp_cont.transform(train_data[continuous])     # transform columns

    imp_cat = SimpleImputer(strategy='constant', missing_values=np.nan, fill_value=0)  # define imputer for categorical variables
    imp_cat = imp_cat.fit(train_data[categorical])                                  # fit imputer on columns
    train_data[categorical] = imp_cat.transform(train_data[categorical])                 # transform columns
    # print(sum(train_data.isnull().any(axis=1)))
    # print(sum(test_data.isnull().any(axis=1)))
    # exit()

    train_data = train_data.values
    train_label = train_label.values
    test_data = test_data.values
    test_label = test_label.values

    # Hyperparameter tuning
    print(f"**********Hyperparameter tuning ***************************")
    print('Params: ', get_search_params(model_name))
    scaler = MinMaxScaler()
    scaler.fit(train_data)
    grid_lr = GridSearchCV(estimator=get_model(model_name), param_grid=get_search_params(model_name), scoring='f1_macro',cv=5, n_jobs=-1, verbose=4)
    grid_lr.fit(scaler.transform(train_data), train_label)

    best_param = grid_lr.best_params_
    train_predict = grid_lr.best_estimator_.predict(scaler.transform(train_data))
    test_predict = grid_lr.best_estimator_.predict(scaler.transform(test_data))

    train_performance = get_performance(model_name, 'best', is_train=True, y_pred=train_predict, y_true=train_label)
    test_performance = get_performance(model_name, 'best', is_train=False, y_pred=test_predict, y_true=test_label)

    # Save performance
    best_params["best"] = best_param
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


# def test_tuning(model_name, target_col):
#      # Initialize paths
#     result_path = os.path.join("../results", target_col, 'impute_' + model_name + "_test.csv")
#     best_param_path = os.path.join("../results", target_col, 'impute_' + model_name + "_best_params_test.json")

#     # data processing
#     x, y = get_dataset(target_col=target_col)

#     best_params = dict()
#     performance = None

#     # Hyperparameter tuning
#     kf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
#     for fold, (train, test) in enumerate(kf.split(x,y)):
#         scaler = MinMaxScaler()
#         scaler.fit(x[train])
#         grid_lr = GridSearchCV(estimator=get_model(model_name), param_grid=get_search_params(model_name), scoring='f1_macro',cv=5)
#         grid_lr.fit(scaler.transform(x[train]), y[train])

#         best_param = grid_lr.best_params_
#         train_predict = grid_lr.best_estimator_.predict(scaler.transform(x[train]))
#         test_predict = grid_lr.best_estimator_.predict(scaler.transform(x[test]))

#         train_performance = get_performance(model_name, fold, is_train=True, y_pred=train_predict, y_true=y[train])
#         test_performance = get_performance(model_name, fold, is_train=False, y_pred=test_predict, y_true=y[test])

#         # Save performance
#         best_params["fold_" + str(fold+1)] = best_param
#         df1 = pd.DataFrame(data=train_performance, index=[0])
#         df2 = pd.DataFrame(data=test_performance, index=[0])
#         if performance is None:
#             performance = pd.concat([df1, df2])
#         else:
#             performance = pd.concat([performance, df1])
#             performance = pd.concat([performance, df2])


#     # Save in files
#     with open(best_param_path, "w") as f:
#         json.dump(best_params, f, indent=2)

#     performance.sort_values(by=["is_train", "fold"], inplace=True)
#     performance.reset_index(drop=True)
#     performance.to_csv(result_path, index=False)
#     print(f"Complete hyperparameter tuning for {model_name} !")


if __name__ == '__main__':
    start_time = datetime.now()
    parser = parse_arguments()
    args = parser.parse_args()
    with parallel_backend('threading', n_jobs=-1):
        tuning(args.model_name, args.target_col)
    end_time = datetime.now()
    print(f"Grid search test for {args.model_name} completes in {end_time - start_time}")

    with open("../logs/log.txt", 'a') as f:
        f.write(f"Grid search test for {args.model_name} completes in {end_time - start_time}\n")
