import argparse
import pandas as pd
import os
from datetime import datetime
import json
from joblib import parallel_backend
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from scipy import stats

from metrics import get_performance
from models import get_model


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help="Input name of a model")
    parser.add_argument('--target_col', type=str, help="Choose thresshold for process", choices=["Severity", "AHI_5", "AHI_15", "AHI_30"])
    parser.add_argument('--imputer', type=str, choices=['mean_const', 'median_const', 'mean_most', 'median_most', 'bayesian', 'rf', 'knn', 'knn2', 'knn5', 'knn10', 'mice1', 'mice2', 'mice3', 'mice4', 'mice5', 'hmisc'])
    parser.add_argument('--outlier', type=str, choices=['dataz', 'iqr'])

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


def tuning(model_name, target_col, imputer, outlier):

    # Initialize paths
    if not os.path.isdir(os.path.join('../results', target_col)):
        os.mkdir(os.path.join('../results', target_col))

    if not os.path.isdir(os.path.join('../results', target_col, model_name)):
        os.mkdir(os.path.join('../results', target_col, model_name))

    result_path = os.path.join("../results", target_col, model_name, f'impute_{imputer}_outlier_{outlier}_' + model_name + "_metrics.csv")
    best_param_path = os.path.join("../results", target_col, model_name, f'impute_{imputer}_outlier_{outlier}_' + model_name + "_best_params.json")

    # data processing
    x, y = get_dataset(target_col=target_col)

    best_params = dict()
    performance = None

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

    # Imputation
    if os.path.isfile(f'../imputer_data/{imputer}_x.npy'):
        print('Load data from imputer_data')
        with open(f'../imputer_data/{imputer}_x.npy', 'rb') as f:
            x = np.load(f)
        with open(f'../imputer_data/{imputer}_y.npy', 'rb') as f:
            y = np.load(f)
    elif imputer == 'median_const':
        
        imp_cont = SimpleImputer(strategy='median', missing_values=np.nan)                 # define imputer for continuous variables
        imp_cont = imp_cont.fit(x[continuous])                                             # fit imputer on columns
        x[continuous] = imp_cont.transform(x[continuous])                                  # transform columns

        imp_cat = SimpleImputer(strategy='constant', missing_values=np.nan, fill_value=0)  # define imputer for categorical variables
        imp_cat = imp_cat.fit(x[categorical])                                              # fit imputer on columns
        x[categorical] = imp_cat.transform(x[categorical])                                 # transform columns

        x = x.values
        y = y.values

    elif imputer == 'median_most':
        imp_cont = SimpleImputer(strategy='median', missing_values=np.nan)                 # define imputer for continuous variables
        imp_cont = imp_cont.fit(x[continuous])                                             # fit imputer on columns
        x[continuous] = imp_cont.transform(x[continuous])                                  # transform columns

        imp_cat = SimpleImputer(strategy='most_frequent', missing_values=np.nan)           # define imputer for categorical variables
        imp_cat = imp_cat.fit(x[categorical])                                              # fit imputer on columns
        x[categorical] = imp_cat.transform(x[categorical])                                 # transform columns
        x = x.values
        y = y.values

    elif imputer == 'mean_const':
        imp_cont = SimpleImputer(strategy='mean', missing_values=np.nan)                   # define imputer for continuous variables
        imp_cont = imp_cont.fit(x[continuous])                                             # fit imputer on columns
        x[continuous] = imp_cont.transform(x[continuous])                                  # transform columns

        imp_cat = SimpleImputer(strategy='constant', missing_values=np.nan, fill_value=0)  # define imputer for categorical variables
        imp_cat = imp_cat.fit(x[categorical])                                              # fit imputer on columns
        x[categorical] = imp_cat.transform(x[categorical])                                 # transform columns
        x = x.values
        y = y.values

    elif imputer == 'mean_most':
        imp_cont = SimpleImputer(strategy='mean', missing_values=np.nan)                   # define imputer for continuous variables
        imp_cont = imp_cont.fit(x[continuous])                                             # fit imputer on columns
        x[continuous] = imp_cont.transform(x[continuous])                                  # transform columns

        imp_cat = SimpleImputer(strategy='most_frequent', missing_values=np.nan)           # define imputer for categorical variables
        imp_cat = imp_cat.fit(x[categorical])                                              # fit imputer on columns
        x[categorical] = imp_cat.transform(x[categorical])                                 # transform columns
        x = x.values
        y = y.values

    elif imputer == 'bayesian':
        imp_mul = IterativeImputer(max_iter=10, random_state=0)
        imp_mul.fit(x)
        x = imp_mul.transform(x)
        y = y.values

    elif imputer == 'rf':
        imp_mult_rf = IterativeImputer(max_iter=10, random_state=0, estimator=RandomForestRegressor())
        imp_mult_rf = imp_mult_rf.fit(x)                          
        x = imp_mult_rf.transform(x)
        y = y.values

    elif imputer == 'knn':
        imp_mult_kn = IterativeImputer(max_iter=10, random_state=0, estimator= KNeighborsRegressor())
        imp_mult_kn = imp_mult_kn.fit(x)                               
        x = imp_mult_kn.transform(x)
        y = y.values

    elif imputer == 'knn2':
        imp_knn2 = KNNImputer(n_neighbors=2, weights="uniform")    
        imp_knn2 = imp_knn2.fit(x)                            
        x = imp_knn2.transform(x)
        y = y.values

    elif imputer == 'knn5':
        imp_knn5 = KNNImputer(n_neighbors=5, weights="uniform")    
        imp_knn5 = imp_knn5.fit(x)                            
        x = imp_knn5.transform(x)
        y = y.values

    elif imputer == 'knn10':
        imp_knn10 = KNNImputer(n_neighbors=10, weights="uniform")    
        imp_knn10 = imp_knn10.fit(x)                            
        x = imp_knn10.transform(x)
        y = y.values
    else:
        raise('Incorrect Imputer')

    if not os.path.isfile(f'../imputer_data/{imputer}_x.npy'):
        with open(f'../imputer_data/{imputer}_x.npy', 'wb') as f:
            np.save(f, x)

        with open(f'../imputer_data/{imputer}_y.npy', 'wb') as f:
            np.save(f, y)

    # Outlier processing
    data = pd.DataFrame(data=x, columns=features)
    data['Severity'] = y
    continuous_z = ['Age','Height','Weight','Cervical_perimeter','Abdominal_perimeter','Systolic_BP','Diastolic_BP', 
              'BMI','Epworth_scale','Pichots_scale','Depression_scale', 'Severity']
    
    if outlier == 'dataz':
        z = data[continuous_z].groupby(['Severity']).transform(stats.zscore).abs()
        data_z = data[(z < 3).all(axis=1)]
        x = data_z[features].values
        y = data_z['Severity'].values

    elif outlier == 'iqr':
        def quartile(df):
            Q1 = df[continuous_z].quantile(0.25) # Calculating quartile 1 by severity
            Q3 = df[continuous_z].quantile(0.75)
            IQR = Q3 - Q1
            df= df[~((df[continuous_z] < (Q1 - 1.5 * IQR)) |(df[continuous_z] > (Q3 + 1.5 * IQR))).any(axis=1)]
            return df

        data_IQR = data.groupby('Severity').apply(quartile).reset_index(level=0, drop=True).sort_index()
        x = data_IQR[features].values
        y = data_IQR['Severity'].values

    else:
        raise('Invalid outlier')

    print('Data shape: ', x.shape)
    # Hyperparameter tuning
    kf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
    for fold, (train, test) in enumerate(kf.split(x,y)):
        print(f"**********Hyperparameter tuning for fold {fold}***************************")
        print('Params: ', get_search_params(model_name))
        scaler = MinMaxScaler()
        scaler.fit(x[train])
        grid_lr = GridSearchCV(estimator=get_model(model_name), param_grid=get_search_params(model_name), scoring='f1_macro',cv=5, n_jobs=-1, verbose=4)
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


def test_tuning(model_name, target_col):
     # Initialize paths
    result_path = os.path.join("../results", target_col, model_name + "_test.csv")
    best_param_path = os.path.join("../results", target_col, model_name + "_best_params_test.json")

    # data processing
    x, y = get_dataset(target_col=target_col)

    best_params = dict()
    performance = None

    # Hyperparameter tuning
    kf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
    for fold, (train, test) in enumerate(kf.split(x,y)):
        scaler = MinMaxScaler()
        scaler.fit(x[train])
        grid_lr = GridSearchCV(estimator=get_model(model_name), param_grid=get_search_params(model_name), scoring='f1_macro',cv=5)
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
    with parallel_backend('threading', n_jobs=-1):
        tuning(args.model_name, args.target_col, args.imputer, args.outlier)
    end_time = datetime.now()
    print(f"Grid search test for {args.model_name}_{args.imputer}_{args.outlier} completes in {end_time - start_time}")

    with open("../logs/log.txt", 'a') as f:
        f.write(f"Grid search test for {args.model_name}_{args.imputer}_{args.outlier} completes in {end_time - start_time}\n")
