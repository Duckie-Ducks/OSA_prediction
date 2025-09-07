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
from sklearn.model_selection import StratifiedKFold
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

import os
import pickle
import numpy as np

#Add Retrain mode dung best params -> Train metrics

def parse_arguments():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model', type=str, help='Model type, current support GANDALF, FTT, TabNet, TabTransformer')
    parser.add_argument('--target_col', type=str, help='The column of the target label, choose Severity for multi-class, AHI_5 for binary cut-off at 5')
    parser.add_argument('--imb', type=str, help='Imbalance handling strategy, choose ADASYN or SMOTE otherwise None', default=256)
    parser.add_argument('--features_name', type=str, help="Choose collection of features for learning", choices=["all_features", "Demographic", "Measurements", "Symptoms", "Questionnaires", 'Comorbidities', "Mencar", "Wu", "Huang", "Rodruiges", "Ustun"])
    parser.add_argument('--task', type=str, help='Imputation handling strategy', default='median_const')
    return parser

def load_OSA(args):
    path = 'data/OSA_complete_AHI.csv'

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

#from sklearn.metrics import root_mean_squared_error
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

parser = parse_arguments()
args = parser.parse_args()
print(args)
data, cat_col_names, num_col_names, target_col = load_OSA(args)




model_configs = {'GANDALF':GANDALFConfig(task="classification"),
                'FTT':FTTransformerConfig(task="classification"), 
                'TabNet':TabNetModelConfig(task="classification"), 
                'TabTransformer':TabTransformerConfig(task="classification"),
                'CEM': CategoryEmbeddingModelConfig(task="classification"), 
                'GATE':GatedAdditiveTreeEnsembleConfig(task="classification"), 
                'Node':NodeConfig(task="classification"), 
                'DANet':DANetConfig(task="classification"), 
                'AutoInt':AutoIntConfig(task="classification")}

sss = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)

#model_path = 'results_all_feature/{}/{}'.format(args.target_col ,args.model)
#model_path = 'results_imb/{}/{}'.format(args.target_col ,args.model)
#model_path = 'results_imb/{}/{}/{}/{}'.format(args.target_col, args.imb, args.features_name ,args.model)
if args.task == "classification":
    target_col = args.target_col
    model_path = 'results/{}/{}/{}/{}'.format(args.target_col, args.imp, args.imb, args.model)
    df_result = pd.DataFrame(columns=['model', 'accuracy', 'recall_weighted', 'recall_macro', 'f1_weighted', 'f1_macro', 'bal_acc', 'precision_weighted', 'precision_macro', 'g_mean'])
else:
    target_col = 'AHI'
    model_path = 'results_regression/{}/{}/{}/{}'.format(target_col, 'median_const', 'None', args.model)
    df_result = pd.DataFrame(columns=['model', 'MSE', 'RMSE', 'MAPE', 'MAPE_nonzero'])

#severity_to_ahi = {0: 0, 1: 5, 2: 15, 3: 30}
import scipy
#data['AHI'] = data['Severity'].map(severity_to_ahi)
y_label = data[args.target_col]
data = data.drop(args.target_col, axis=1)
import numpy as np
fold_idx = 1
for train_ids, test_ids in sss.split(data, y_label):
    print('Eval fold: ', fold_idx)
    train = data.iloc[train_ids]
    test =  data.iloc[test_ids]
    best_model_path = os.path.join(model_path, 'best_model_fold_{}.pkl'.format(fold_idx))
    #Loading the model from the saved file
    with open(best_model_path, 'rb') as file:
        model = pickle.load(file)
    pred_df = model.predict(test)
    preds = pred_df['AHI_prediction']
    pred_df_train = model.predict(train)
    preds_train = pred_df_train['AHI_prediction']
    scipy.io.savemat('test_actual_{}.mat'.format(fold_idx), {'data_actual_21': test[target_col]})
    scipy.io.savemat('test_preds_{}.mat'.format(fold_idx), {'data_preds_21': preds})
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
