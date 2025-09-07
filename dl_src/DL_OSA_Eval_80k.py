from pytorch_tabular.models import GANDALFConfig, FTTransformerConfig, TabNetModelConfig, TabTransformerConfig
from pytorch_tabular import TabularModel
from pytorch_tabular.models import (
    GANDALFConfig, 
    FTTransformerConfig, 
    TabNetModelConfig, 
    TabTransformerConfig, 
    CategoryEmbeddingModelConfig, 
    GatedAdditiveTreeEnsembleConfig, 
    NodeConfig, 
    DANetConfig, 
    AutoIntConfig
)
from sklearn.model_selection import train_test_split
from pytorch_tabular import available_models
import pandas as pd
import json
import warnings
from sklearn.model_selection import KFold, StratifiedKFold
from pytorch_tabular.tabular_model_tuner import TabularModelTuner
import argparse
from pathlib import Path

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import make_scorer
from sklearn.metrics import balanced_accuracy_score
from imblearn.metrics import geometric_mean_score
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import os
import pickle
import numpy as np

#Add Retrain mode dung best params -> Train metrics

def parse_arguments():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model', type=str, help='Model type, current support GANDALF, FTT, TabNet, TabTransformer')
    parser.add_argument('--target_col', type=str, help='The column of the target label, choose Severity for multi-class, AHI_5 for binary cut-off at 5')
    parser.add_argument('--imb', type=str, help='Imbalance handling strategy, choose ADASYN or SMOTE', default='None')
    parser.add_argument('--imp', type=str, help='Imputation handling strategy', default='None')
    parser.add_argument('--task', type=str, help='Imputation handling strategy', default='None')
    parser.add_argument('--features_name', type=str, help="Choose collection of features for learning", choices=["all_features", "Demographic", "Measurements", "Symptoms", "Questionnaires", 'Comorbidities', "Mencar", "Wu", "Huang", "Rodruiges", "Ustun"])
    return parser

def load_OSA(args):
    path = 'data/OSA_complete_patients.csv'

    df= pd.read_csv(path, index_col = ['PatientID'])
    df.drop(df.columns[[0]], axis=1, inplace=True)
    df.head(5)

    if args.features_name is not None:
        select_fea = get_feature(args.features_name) +['Severity']
        all_header = list(df)
        drop_feature = set(select_fea) ^ set(all_header) #Drop unused features
        df.drop(drop_feature, axis=1, inplace=True)

   
    cutoff_val = 10 # if there are more than 10 unique vals, we consider as continous feature
    cat_cols = []
    cont_cols = []
    for col in df.columns[:-1]:
        if len(df[col].unique()) > cutoff_val:
            cont_cols.append(col)
        else:
            cat_cols.append(col)
    # I am going to add columns AHI5, AHI15, and AHI30 
    df['AHI_5'] = df['Severity'].apply(lambda x: 1 if x >= 1 else 0)
    df['AHI_15'] = df['Severity'].apply(lambda x: 1 if x >= 2 else 0)
    df['AHI_30'] = df['Severity'].apply(lambda x: 1 if x >= 3 else 0)
    return df, cat_cols, cont_cols, args.target_col

def load_OSA_80k_fixed(args):
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

    path = 'data/OSA_classifier_AHI.csv'
    df= pd.read_csv(path)
    df.head()
    x = df.copy(deep= True)
    x = x[features]
    # I am going to add columns AHI5, AHI15, and AHI30 
    df['AHI_5'] = df['Severity'].apply(lambda x: 1 if x >= 1 else 0)
    df['AHI_15'] = df['Severity'].apply(lambda x: 1 if x >= 2 else 0)
    df['AHI_30'] = df['Severity'].apply(lambda x: 1 if x >= 3 else 0)
    
    # if args.task == "classification":
    y = df[args.target_col]
    imputer = args.imp

    other_imp = False
    # Imputation
    if imputer == 'median_const':
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
    elif imputer == "mice_imp1_maxit20":
        df= pd.read_csv(path)
        df.head()
        x = df.copy(deep= True)
        x = x[features]
        # I am going to add columns AHI5, AHI15, and AHI30 
        df['AHI_5'] = df['Severity'].apply(lambda x: 1 if x >= 1 else 0)
        df['AHI_15'] = df['Severity'].apply(lambda x: 1 if x >= 2 else 0)
        df['AHI_30'] = df['Severity'].apply(lambda x: 1 if x >= 3 else 0)
        other_imp = True
    elif imputer == "mice_imp1_maxit20":
        df= pd.read_csv('data/mice_imp1_maxit20.csv', index_col=0)
        df.head()
        x = df.copy(deep= True)
        x = x[features]
        # I am going to add columns AHI5, AHI15, and AHI30 
        df['AHI_5'] = df['Severity'].apply(lambda x: 1 if x >= 1 else 0)
        df['AHI_15'] = df['Severity'].apply(lambda x: 1 if x >= 2 else 0)
        df['AHI_30'] = df['Severity'].apply(lambda x: 1 if x >= 3 else 0)
        other_imp = True
    elif imputer == "mice_imp2_maxit20":
        df= pd.read_csv('data/mice_imp2_maxit20.csv')
        df.head()
        x = df.copy(deep= True)
        x = x[features]
        # I am going to add columns AHI5, AHI15, and AHI30 
        df['AHI_5'] = df['Severity'].apply(lambda x: 1 if x >= 1 else 0)
        df['AHI_15'] = df['Severity'].apply(lambda x: 1 if x >= 2 else 0)
        df['AHI_30'] = df['Severity'].apply(lambda x: 1 if x >= 3 else 0)
        other_imp = True
    elif imputer == "mice_imp3_maxit20":
        df= pd.read_csv('data/mice_imp3_maxit20.csv')
        df.head()
        x = df.copy(deep= True)
        x = x[features]
        # I am going to add columns AHI5, AHI15, and AHI30 
        df['AHI_5'] = df['Severity'].apply(lambda x: 1 if x >= 1 else 0)
        df['AHI_15'] = df['Severity'].apply(lambda x: 1 if x >= 2 else 0)
        df['AHI_30'] = df['Severity'].apply(lambda x: 1 if x >= 3 else 0)
        other_imp = True
    elif imputer == "mice_imp4_maxit20":
        df= pd.read_csv('data/mice_imp4_maxit20.csv')
        df.head()
        x = df.copy(deep= True)
        x = x[features]
        # I am going to add columns AHI5, AHI15, and AHI30 
        df['AHI_5'] = df['Severity'].apply(lambda x: 1 if x >= 1 else 0)
        df['AHI_15'] = df['Severity'].apply(lambda x: 1 if x >= 2 else 0)
        df['AHI_30'] = df['Severity'].apply(lambda x: 1 if x >= 3 else 0)
        other_imp = True
    elif imputer == "mice_imp5_maxit20":
        df= pd.read_csv('data/mice_imp5_maxit20.csv')
        df.head()
        x = df.copy(deep= True)
        x = x[features]
        # I am going to add columns AHI5, AHI15, and AHI30 
        df['AHI_5'] = df['Severity'].apply(lambda x: 1 if x >= 1 else 0)
        df['AHI_15'] = df['Severity'].apply(lambda x: 1 if x >= 2 else 0)
        df['AHI_30'] = df['Severity'].apply(lambda x: 1 if x >= 3 else 0)
        other_imp = True
    elif imputer == "completed_hmisc_impute1":
        res_df= pd.read_csv('data/completed_hmisc_impute1.csv')
        res_df.head()
        #x = df.copy(deep= True)
        #print(features)
        #exit()
        print(x)
        #exit()
        #x = x[features]
        for fea in features:
            print(fea)
            x[fea] = res_df[fea]

        # I am going to add columns AHI5, AHI15, and AHI30 
        df['AHI_5'] = df['Severity'].apply(lambda x: 1 if x >= 1 else 0)
        df['AHI_15'] = df['Severity'].apply(lambda x: 1 if x >= 2 else 0)
        df['AHI_30'] = df['Severity'].apply(lambda x: 1 if x >= 3 else 0)
        other_imp = True
    else:
        raise('Incorrect Imputer')
    
    if other_imp:
        y = df[args.target_col]
        return x,y, df
    idx = 0
    for fea in features:
        df[fea] = x[:,idx]
        idx += 1

    return x,y, df, continuous, categorical

def get_feature(feature_name):
    with open("config/Feature_Selection/feature_config.json") as f:
        feature_collections = json.load(f)

    return feature_collections[feature_name]

def eval_metric(test, preds, args, type):
    # Calculate metrics and update DataFrame
    metrics = {
        'accuracy': accuracy_score(test, preds),
        'recall_macro': recall_score(test, preds, average='macro'),
        'recall_weighted': recall_score(test, preds, average='weighted'),
        'f1_weighted': f1_score(test, preds, average='weighted'),
        'f1_macro': f1_score(test, preds, average='macro'),
        'bal_acc': balanced_accuracy_score(test, preds),
        'precision_weighted': precision_score(test, preds, average='weighted'),
        'precision_macro': precision_score(test, preds, average='macro'),
        'g_mean': geometric_mean_score(test, preds)
    }

    row = {'model': args.model + str(type),
           'accuracy': round(metrics['accuracy'], 3),
           'recall_macro': round(metrics['recall_macro'], 3),
           'recall_weighted': round(metrics['recall_weighted'], 3),
           'f1_weighted': round(metrics['f1_weighted'], 3),
           'f1_macro': round(metrics['f1_macro'], 3),
           'bal_acc': round(metrics['bal_acc'], 3),
           'precision_weighted': round(metrics['precision_weighted'], 3),
           'precision_macro': round(metrics['precision_macro'], 3),
           'g_mean': round(metrics['g_mean'], 3)}

    return row


parser = parse_arguments()
args = parser.parse_args()
print(args)
#data, cat_col_names, num_col_names, target_col = load_OSA(args)
x, y, data, cat_col_names, num_col_names, = load_OSA_80k_fixed(args)




model_configs = {'GANDALF':GANDALFConfig(task=args.task),
                'FTT':FTTransformerConfig(task=args.task), 
                'TabNet':TabNetModelConfig(task=args.task), 
                'TabTransformer':TabTransformerConfig(task=args.task),
                'CEM': CategoryEmbeddingModelConfig(task=args.task), 
                'GATE':GatedAdditiveTreeEnsembleConfig(task=args.task), 
                'Node':NodeConfig(task=args.task), 
                'DANet':DANetConfig(task=args.task), 
                'AutoInt':AutoIntConfig(task=args.task)}

#sss = KFold(n_splits=5, random_state=0, shuffle=True)
kf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
from math import sqrt

def calculate_mape(true_labels, predicted_labels):
    """
    Calculate the Mean Absolute Percentage Error (MAPE) excluding samples where true_labels are zero.

    Parameters:
    true_labels (array-like): Array of true (actual) values.
    predicted_labels (array-like): Array of predicted values.

    Returns:
    float: The MAPE value.
    """
    # Convert inputs to numpy arrays for easier manipulation
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)
    
    # Exclude samples where true_labels are zero
    non_zero_mask = true_labels != 0
    true_labels_non_zero = true_labels[non_zero_mask]
    predicted_labels_non_zero = predicted_labels[non_zero_mask]
    
    # Calculate the absolute percentage error for each sample
    absolute_percentage_errors = np.abs((true_labels_non_zero - predicted_labels_non_zero) / true_labels_non_zero)
    
    # Calculate the mean of the absolute percentage errors
    mape = np.mean(absolute_percentage_errors) * 100  # Multiply by 100 to get percentage
    
    return mape

def eval_metric_regression(test, preds, args, type):
    # Calculate metrics and update DataFrame
    metrics = {
        'MSE': mean_squared_error(test, preds),
        'RMSE': sqrt(mean_squared_error(test, preds)),
        'MAPE': mean_absolute_percentage_error(test, preds),
        'MAPE_nonzero': calculate_mape(test, preds),
        'MAE': mean_absolute_error(test, preds)
    }

    row = {'model': args.model + str(type),
           'MSE': round(metrics['MSE'], 3),
           'RMSE': round(metrics['RMSE'], 3),
           'MAPE': round(metrics['MAPE'], 3),
            'MAPE_nonzero': round(metrics['MAPE_nonzero'], 3),
            'MAE': round(metrics['MAE'], 3)
           }

    return row
#model_path = 'results_all_feature/{}/{}'.format(args.target_col ,args.model)
#model_path = 'results_imb/{}/{}'.format(args.target_col ,args.model)
#model_path = 'results_imb/{}/{}/{}/{}'.format(args.target_col, args.imb, args.features_name ,args.model)
if args.task == "classification":
    model_path = 'results_80k/{}/{}/{}/{}'.format(args.target_col, args.imp, args.imb, args.model)
    df_result = pd.DataFrame(columns=['model', 'accuracy', 'recall_weighted', 'recall_macro', 'f1_weighted', 'f1_macro', 'bal_acc', 'precision_weighted', 'precision_macro', 'g_mean'])
else:
    target_col = 'AHI'
    model_path = 'results_80k_regression/{}/{}/{}/{}'.format(target_col, args.imp, 'None', args.model)
    df_result = pd.DataFrame(columns=['model', 'MSE', 'RMSE', 'MAPE', 'MAPE_nonzero'])

import numpy as np
import scipy
#fold_idx = 1
#for train_ids, test_ids in sss.split(data):
for fold_idx, (train_ids, test_ids) in enumerate(kf.split(x,y)):
    target_col = 'AHI'
    print('Eval fold: ', fold_idx)
    train = data.iloc[train_ids]
    test =  data.iloc[test_ids]
    best_model_path = os.path.join(model_path, 'best_model_fold_{}.pkl'.format(fold_idx))
    print(best_model_path)
    #Loading the model from the saved file
    with open(best_model_path, 'rb') as file:
        model = pickle.load(file)
    pred_df = model.predict(test)
    preds = pred_df['AHI_prediction']
    pred_df_train = model.predict(train)
    preds_train = pred_df_train['AHI_prediction']
    print(test[target_col][:10])
    print(preds[:10])
    print(test[target_col][:10])
    print(preds[:10])
    print(len(test[target_col])-np.count_nonzero(test[target_col]))
    scipy.io.savemat('test_actual_{}.mat'.format(fold_idx), {'data_actual_80': test[target_col]})
    scipy.io.savemat('test_preds_{}.mat'.format(fold_idx), {'data_preds_80': preds})
    if args.task == "regression":
        row_test = eval_metric_regression(test[target_col], preds, args, type ='test')
        row_train= eval_metric_regression(train[target_col], preds_train, args, type ='train')
    else:
        row_test = eval_metric(test[target_col], preds, args, type ='test')
        row_train= eval_metric(train[target_col], preds_train, args, type ='train')
    df_result = pd.concat([df_result, pd.DataFrame([row_test])], ignore_index=True)
    df_result = pd.concat([df_result, pd.DataFrame([row_train])], ignore_index=True)
    #Predict
    fold_idx +=1
    pred_path = os.path.join(model_path,'pred_{}.npy'.format(fold_idx))
    test_ids_path = os.path.join(model_path,'test_ids_{}.npy'.format(fold_idx))
    np.save(pred_path, preds)
    np.save(test_ids_path, test_ids)

result_path = os.path.join(model_path,'full_metric_result.csv'.format(fold_idx))
df_result.to_csv(result_path, index = False)
