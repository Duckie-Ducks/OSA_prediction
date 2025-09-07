import argparse
import pandas as pd
import os
from datetime import datetime
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from imblearn.over_sampling import ADASYN, SMOTE
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
from scipy import stats

from metrics import get_performance
from models import get_model
from utils import MatrixExcelSaver, create_folder


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


def get_param(model_name, fold, target_col, imputer, outlier):
    data_path = os.path.join("../results", target_col, model_name, f'impute_{imputer}_outlier_{outlier}_'+model_name + '_best_params.json')
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

def feature_selection(model_name, target_col, balance_strategy, features_name, imputer, outlier):
    result_path = os.path.join('../results', target_col, model_name, f'impute_{imputer}_outlier_{outlier}_'+model_name+'_'+balance_strategy+'_'+features_name+'.csv')

    features = get_feature(feature_name=features_name)
    performance = None
    with open(f'../imputer_data/{imputer}_x.npy', 'rb') as f:
        x = np.load(f)

    with open(f'../imputer_data/{imputer}_y.npy', 'rb') as f:
        y = np.load(f)

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
        # Normalize data
        scaler = MinMaxScaler()
        scaler.fit(x[train])
        x_scale = scaler.transform(x)

        # Model initialize weights
        model = get_model(model_name)
        param = get_param(model_name, fold+1, target_col, imputer, outlier)

        model.set_params(**param)

        # Balancing
        balancer = get_balance_strategy(balance_strategy)

        if balancer is not None:
            X_train_balanced, y_train_balanced = balancer.fit_resample(x_scale[train], y[train])
            model.fit(X_train_balanced, y_train_balanced)
        else:
            model.fit(x_scale[train], y[train])

        train_predict = model.predict(x_scale[train])
        test_predict = model.predict(x_scale[test])

        train_performance = get_performance(model_name, fold, is_train=True, y_pred=train_predict, y_true=y[train])
        test_performance = get_performance(model_name, fold, is_train=False, y_pred=test_predict, y_true=y[test])

        # Save confusion matrix
        test_conf = confusion_matrix(y[test], test_predict)
        train_conf = confusion_matrix(y[train], train_predict)
        
        create_folder(os.path.join('../results', target_col, model_name))
        train_conf_result_path = os.path.join('../results', target_col, model_name, model_name+'_'+balance_strategy+'_'+features_name+'_'+'fold'+str(fold+1)+'_'+'train'+'.xlsx')
        test_conf_result_path = os.path.join('../results', target_col, model_name, model_name+'_'+balance_strategy+'_'+features_name+'_'+'fold'+str(fold+1)+'_'+'test'+'.xlsx')
        MatrixExcelSaver.save(test_conf, test_conf_result_path)
        MatrixExcelSaver.save(train_conf, train_conf_result_path)

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
    print(f"Complete feature selection for {model_name} - {balance_strategy} - {features_name} - {balance_strategy} - {imputer} - {outlier}")

    pass

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help="Input name of a model", choices=["GaussianNB", "LinearSVC", "RandomForestClassifier", "LGBMClassifier", "LogisticRegression", "ExtraTreesClassifier", "LinearDiscriminantAnalysis", "XGBClassifier", "CatBoostClassifier", 'KNeighborsClassifier', 'SVC', 'SVM', 'ExtraTreesClassifier', 'HistGradientBoostingClassifier'])
    parser.add_argument('--target_col', type=str, help="Choose thresshold for process", choices=["Severity", "AHI_5", "AHI_15", "AHI_30"])
    parser.add_argument('--balance_strategy', type=str, help="Choose strategy for balancing dataset", choices=["adasyn", "smote", "None", "borderline", "underline"])
    parser.add_argument('--features_name', type=str, help="Choose collection of features for learning", choices=["all_features", "Demographic", "Measurements", "Symptoms", "Questionnaires", 'Comorbidities', 'Mencar', 'Huang', 'Rodruiges', 'KBest Chi2', 'RF Impurity', 'RF Permutation', 'KBest Fclass', 'Correlation', 'Kruskall Chi', 'SHAP_RF', 'CatBoost', 'Wu', 'Ustun'])
    parser.add_argument('--imputer', type=str, choices=['mean_const', 'median_const', 'mean_most', 'median_most', 'bayesian', 'rf', 'knn', 'knn2', 'knn5', 'knn10', 'mice1', 'mice2', 'mice3', 'mice4', 'mice5', 'hmisc'])
    parser.add_argument('--outlier', type=str, choices=['dataz', 'iqr'])

    return parser


if __name__ == '__main__':
    start_time = datetime.now()
    parser = parse_arguments()
    args = parser.parse_args()
    feature_selection(args.model_name, args.target_col, args.balance_strategy, args.features_name, args.imputer, args.outlier)
    end_time = datetime.now()
    print(f"Grid search test for {args.model_name}-{args.target_col}-{args.balance_strategy}-{args.features_name}-{args.outlier} completes in {end_time - start_time}")

    with open("../logs/log_balancing.txt", 'a') as f:
        f.write(f"Feature selection for {args.model_name}_{args.imputer}_{args.outlier} completes in {end_time - start_time}\n")
    pass
