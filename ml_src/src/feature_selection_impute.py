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
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer

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


def get_param(model_name, target_col):
    data_path = os.path.join("../results", target_col, model_name, 'impute_hypertuning_' + model_name + '_best_params.json')
    with open(data_path, 'r') as f:
        params = json.load(f)

    return params["best"]


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

def feature_selection(model_name, target_col, balance_strategy, features_name, imputer):
    result_path = os.path.join('../results', target_col, model_name, f'impute_{imputer}_'+model_name+'_'+balance_strategy+'_'+features_name+'.csv')
    features = get_feature(feature_name=features_name)
    # data processing
    x, y = get_dataset(target_col=target_col)

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
        
    # print(set(features).issubset(set(x.columns)))

    # train_data, train_label = x.iloc[train, :], y.iloc[train]
    # test_data, test_label = x.iloc[test, :], y.iloc[test]

    # Imputation
    # if imputer == 'median_const':
    #     imp_cont = SimpleImputer(strategy='median', missing_values=np.nan)  # define imputer for continuous variables
    #     imp_cont = imp_cont.fit(train_data[continuous])                    # fit imputer on columns
    #     train_data[continuous] = imp_cont.transform(train_data[continuous])     # transform columns

    #     imp_cat = SimpleImputer(strategy='constant', missing_values=np.nan, fill_value=0)  # define imputer for categorical variables
    #     imp_cat = imp_cat.fit(train_data[categorical])                                  # fit imputer on columns
    #     train_data[categorical] = imp_cat.transform(train_data[categorical])                 # transform columns
    # elif imputer == 'bayesian':
    imp_mult = IterativeImputer(max_iter=10, random_state=0)
    imp_mult = imp_mult.fit(x)
    train_data = imp_mult.transform(x)
    # print(sum(train_data.isnull().any(axis=1)))
    # print(sum(test_data.isnull().any(axis=1)))
    # exit()

    train_data, train_label = x.iloc[train, :], y.iloc[train]
    test_data, test_label = x.iloc[test, :], y.iloc[test]

    # train_data = train_data.values
    train_label = train_label.values
    test_data = test_data.values
    test_label = test_label.values

    # Normalize data
    scaler = MinMaxScaler()
    scaler.fit(train_data)
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)

    # Model initialize weights
    model = get_model(model_name)
    param = get_param(model_name, target_col)

    model.set_params(**param)

    # Balancing
    balancer = get_balance_strategy(balance_strategy)

    if balancer is not None:
        X_train_balanced, y_train_balanced = balancer.fit_resample(train_data, train_label)
        model.fit(X_train_balanced, y_train_balanced)
    else:
        model.fit(train_data, train_label)

    train_predict = model.predict(train_data)
    test_predict = model.predict(test_data)

    train_performance = get_performance(model_name, 'best', is_train=True, y_pred=train_predict, y_true=y[train])
    test_performance = get_performance(model_name, 'best', is_train=False, y_pred=test_predict, y_true=y[test])

    # Save confusion matrix
    test_conf = confusion_matrix(test_label, test_predict)
    train_conf = confusion_matrix(train_label, train_predict)
    
    create_folder(os.path.join('../results', target_col, model_name))
    train_conf_result_path = os.path.join('../results', target_col, model_name, f'impute_{imputer}_'+model_name+'_'+balance_strategy+'_'+features_name+'_'+'train'+'.xlsx')
    test_conf_result_path = os.path.join('../results', target_col, model_name, f'impute_{imputer}_'+model_name+'_'+balance_strategy+'_'+features_name+'_'+'test'+'.xlsx')
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
    print(f"Complete feature selection for {model_name} - {balance_strategy} - {features_name} - {balance_strategy} !")

    pass

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help="Input name of a model", choices=["GaussianNB", "LinearSVC", "RandomForestClassifier", "LGBMClassifier", "LogisticRegression", "ExtraTreesClassifier", "LinearDiscriminantAnalysis", "XGBClassifier", "CatBoostClassifier", 'KNeighborsClassifier', 'SVC', 'SVM', 'ExtraTreesClassifier', 'HistGradientBoostingClassifier', 'AdaBoostClassifier'])
    parser.add_argument('--target_col', type=str, help="Choose thresshold for process", choices=["Severity", "AHI_5", "AHI_15", "AHI_30"])
    parser.add_argument('--balance_strategy', type=str, help="Choose strategy for balancing dataset", choices=["adasyn", "smote", "None", "borderline", "underline"])
    parser.add_argument('--features_name', type=str, help="Choose collection of features for learning", choices=["all_features", "Demographic", "Measurements", "Symptoms", "Questionnaires", 'Comorbidities', 'Mencar', 'Huang', 'Rodruiges', 'KBest Chi2', 'RF Impurity', 'RF Permutation', 'KBest Fclass', 'Correlation', 'Kruskall Chi', 'SHAP_RF', 'CatBoost', 'Wu', 'Ustun'])
    parser.add_argument('--imputer', type=str, default='median_const', choices=['median_const', 'bayesian'])

    return parser


if __name__ == '__main__':
    start_time = datetime.now()
    parser = parse_arguments()
    args = parser.parse_args()
    feature_selection(args.model_name, args.target_col, args.balance_strategy, args.features_name, args.imputer)
    end_time = datetime.now()
    print(f"Grid search test for {args.model_name}-{args.target_col}-{args.balance_strategy}-{args.features_name} completes in {end_time - start_time}")

    with open("../logs/log_balancing.txt", 'a') as f:
        f.write(f"Feature selection for {args.model_name} completes in {end_time - start_time}\n")
    pass
