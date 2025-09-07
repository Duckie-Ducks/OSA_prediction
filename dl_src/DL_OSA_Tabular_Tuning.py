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
from pytorch_tabular import TabularModel
from pytorch_tabular.config import (
    DataConfig,
    OptimizerConfig,
    TrainerConfig,
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
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import numpy as np
import os
from datetime import datetime
import pickle

def parse_arguments():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model', type=str, help='Model type, current support GANDALF, FTT, TabNet, TabTransformer')
    parser.add_argument('--target_col', type=str, help='The column of the target label, choose Severity for multi-class, AHI_5 for binary cut-off at 5')
    parser.add_argument('--batch_size', type=int, help='batch_size', default=256)
    parser.add_argument('--task', type=str, help='batch_size', default='classification')
    parser.add_argument('--imb', type=str, help='Imbalance handling strategy, choose ADASYN or SMOTE', default='None')
    parser.add_argument('--imp', type=str, help='Imputation handling strategy', default='None')
    parser.add_argument('--uncleaned_data', type=int, help='Imp handling strategy, 1: Mean/0, 2:Mean/Most Freq, 3:Median/0, 4:Median/Most Freq', default=0)
    
    return parser

def F1(y_true, y_pred):
    return f1_score(y_true, y_pred['prediction'], average='macro')


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

def load_OSA_80k(args, target_col):
    if args.uncleaned_data:
        path = 'data/OSA_classifier.csv'
        df= pd.read_csv(path)
        df.head()
        print(df.shape)
        #exit()
    else:
        path = 'data/OSA_complete_AHI.csv'

        df= pd.read_csv(path, index_col = ['PatientID'])
        df.drop(df.columns[[0]], axis=1, inplace=True)
        df.head(5)

        select_fea = get_feature(args.features_name) +['Severity']
        all_header = list(df)
        drop_feature = set(select_fea) ^ set(all_header) #Drop unused features
        df.drop(drop_feature, axis=1, inplace=True)
   
    cutoff_val = 10 # if there are more than 10 unique vals, we consider as continous feature
    '''
    cat_cols = []
    cont_cols = []
    for col in df.columns[:-1]:
        if len(df[col].unique()) > cutoff_val:
            cont_cols.append(col)
        else:
            cat_cols.append(col)
    '''
    cont_cols = ['Age','Height','Weight','Cervical_perimeter','Abdominal_perimeter','Systolic_BP','Diastolic_BP', 
              'BMI','Epworth_scale','Pichots_scale','Depression_scale']
    cat_cols = [i for i in features if i not in cont_cols]

    mean0 = df.copy(deep= True)           # copy original df
    
    print(args.imp)
    if args.imp == '1':
        print('ZZZZ')
        imp_cont = SimpleImputer(strategy='mean', missing_values=np.nan)  # define imputer for continuous variables
        imp_cat = SimpleImputer(strategy='constant', missing_values=np.nan, fill_value = 0)  # define imputer for categorical variables
    elif args.imp == '2':
        imp_cont = SimpleImputer(strategy='mean', missing_values=np.nan)  # define imputer for continuous variables
        imp_cat = SimpleImputer(strategy='most_frequent', missing_values=np.nan, fill_value = 0)  # define imputer for categorical variables
    elif args.imp == '3':
        imp_cont = SimpleImputer(strategy='median', missing_values=np.nan)  # define imputer for continuous variables
        imp_cat = SimpleImputer(strategy='constant', missing_values=np.nan, fill_value = 0)  # define imputer for categorical variables
    elif args.imp == '4':
        imp_cont = SimpleImputer(strategy='median', missing_values=np.nan)  # define imputer for continuous variables
        imp_cat = SimpleImputer(strategy='most_frequent', missing_values=np.nan, fill_value = 0)  # define imputer for categorical variables
    
    print(imp_cont)
    if args.imp != '0':
        imp_cont = imp_cont.fit(mean0[cont_cols])                    # fit imputer on columns
        mean0[cont_cols] = imp_cont.transform(mean0[cont_cols])     # transform columns
        imp_cat = imp_cat.fit(mean0[cat_cols])                                  # fit imputer on columns
        mean0[cat_cols] = imp_cat.transform(mean0[cat_cols])
        df = mean0

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
    if args.task == "classification":
        y = df[args.target_col]
    else:
        y = df['Severity']

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
        print('No Imputer')
    
    if other_imp:
        y = df[args.target_col]
        return x,y, df
    print(imputer)
    if imputer != 'None':
        idx = 0
        for fea in features:
            df[fea] = x[:,idx]
            idx += 1

    return x,y, df, continuous, categorical

def load_OSA(target_col):
    path = 'data/OSA_complete_AHI.csv'

    df= pd.read_csv(path, index_col = ['PatientID'])
    df.drop(df.columns[[0]], axis=1, inplace=True)
    df.head(5)
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
    return df, categorical, continuous, target_col

#data, cat_col_names, num_col_names, target_col = load_covertype_dataset()
start_time = datetime.now()
parser = parse_arguments()
args = parser.parse_args()
print(args)
    

model_configs = {'GANDALF':GANDALFConfig(task=args.task),
                'FTT':FTTransformerConfig(task=args.task), 
                'TabNet':TabNetModelConfig(task=args.task), 
                'TabTransformer':TabTransformerConfig(task=args.task),
                'CEM': CategoryEmbeddingModelConfig(task=args.task), 
                'GATE':GatedAdditiveTreeEnsembleConfig(task=args.task), 
                'Node':NodeConfig(task=args.task), 
                'DANet':DANetConfig(task=args.task), 
                'AutoInt':AutoIntConfig(task=args.task)}

model_config = model_configs[args.model]

sss = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
# Mapping dictionary
#severity_to_ahi = {0: 0, 1: 5, 2: 15, 3: 30}


with open('config/{}.json'.format(args.model), "r") as f:
    search_space = json.load(f)

if args.uncleaned_data:
    x, y, data, cat_col_names, num_col_names, = load_OSA_80k_fixed(args)
    # Add 'AHI' column by mapping 'Severity' values
    # Hyperparameter tuning for imputer
    if args.task == "classification":
        target_col = args.target_col
        model_path = 'results_80k/{}/{}/{}/{}'.format(args.target_col, args.imp, args.imb, args.model)
    else:
        target_col = 'AHI'
        model_path = 'results_80k_regression/{}/{}/{}/{}'.format(target_col, args.imp, args.imb, args.model)
    Path(model_path).mkdir(parents=True, exist_ok=True)
    
    kf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)

    for fold_idx, (train_ids, test_ids) in enumerate(kf.split(x,y)):
        train = data.iloc[train_ids]
        test =  data.iloc[test_ids]
        data_config = DataConfig(
            target=[
                target_col
            ],  # target should always be a list. Multi-targets are only supported for regression. Multi-Task Classification is not implemented
            continuous_cols=num_col_names,
            categorical_cols=cat_col_names,
        )
        trainer_config = TrainerConfig(
            batch_size=args.batch_size,
            max_epochs=150,
            checkpoints_every_n_epochs = 150,
            load_best = False,
        )

        optimizer_config = OptimizerConfig()

        tuner = TabularModelTuner(
            data_config=data_config,
            model_config=model_config,
            optimizer_config=optimizer_config,
            trainer_config=trainer_config
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if args.task == "classification":
                result = tuner.tune(
                    train=train,
                    validation=test,
                    search_space=search_space,
                    strategy="grid_search",
                    #cv=5, # Uncomment this to do a 5 fold cross validation 
                    metric=F1, # "mean_squared_error"
                    mode="max",
                    progress_bar=True,
                    verbose=False # Make True if you want to log metrics and params each iteration
                )
            else:
                result = tuner.tune(
                    train=train,
                    validation=test,
                    search_space=search_space,
                    strategy="grid_search",
                    #cv=5, # Uncomment this to do a 5 fold cross validation 
                    metric='mean_squared_error', # "mean_squared_error"
                    mode="max",
                    progress_bar=True,
                    verbose=False # Make True if you want to log metrics and params each iteration
                )
            with open(os.path.join(model_path, 'best_params_fold_{}.json'.format(fold_idx)), 'w') as f:
                json.dump(result.best_params, f)
            result.trials_df.to_csv(os.path.join(model_path,'log_fold_{}.csv'.format(fold_idx)))
            # Now, let's save this model to a file
            with open(os.path.join(model_path, 'best_model_fold_{}.pkl'.format(fold_idx)), 'wb') as file:
                pickle.dump(result.best_model, file)

    print('Grid search complete in: ', (datetime.now() - start_time))
    exit()
    #Old training

data, cat_col_names, num_col_names, target_col = load_OSA(target_col=args.target_col)


fold_idx = 1
y_label = data[args.target_col]

if args.task == "classification":
    target_col = args.target_col
    model_path = 'results/{}/{}/{}/{}'.format(args.target_col, args.imp, args.imb, args.model)
else:
    target_col = 'AHI'
    model_path = 'results_regression/{}/{}/{}/{}'.format(target_col, args.imp, args.imb, args.model)
    print(data.columns)
    data = data.drop(args.target_col, axis = 1)
Path(model_path).mkdir(parents=True, exist_ok=True)

print(target_col)
for train_ids, test_ids in sss.split(data, y_label):
    train = data.iloc[train_ids]
    test =  data.iloc[test_ids]
    data_config = DataConfig(
        target=[
            target_col
        ],  # target should always be a list. Multi-targets are only supported for regression. Multi-Task Classification is not implemented
        continuous_cols=num_col_names,
        categorical_cols=cat_col_names,
    )
    print(cat_col_names)
    trainer_config = TrainerConfig(
        batch_size=args.batch_size,
        max_epochs=150,
        checkpoints_every_n_epochs = 150,
        load_best = False,
    )

    optimizer_config = OptimizerConfig()

    tuner = TabularModelTuner(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if args.task == "classification":
            result = tuner.tune(
                train=train,
                validation=test,
                search_space=search_space,
                strategy="grid_search",
                #cv=5, # Uncomment this to do a 5 fold cross validation 
                metric=F1, # "mean_squared_error"
                mode="max",
                progress_bar=True,
                verbose=False # Make True if you want to log metrics and params each iteration
            )
        else:
            result = tuner.tune(
                train=train,
                validation=test,
                search_space=search_space,
                strategy="grid_search",
                #cv=5, # Uncomment this to do a 5 fold cross validation 
                metric='mean_squared_error', # "mean_squared_error"
                mode="max",
                progress_bar=True,
                verbose=False # Make True if you want to log metrics and params each iteration
            )
        with open(os.path.join(model_path, 'best_params_fold_{}.json'.format(fold_idx)), 'w') as f:
            json.dump(result.best_params, f)
        result.trials_df.to_csv(os.path.join(model_path,'log_fold_{}.csv'.format(fold_idx)))
        # Now, let's save this model to a file
        with open(os.path.join(model_path, 'best_model_fold_{}.pkl'.format(fold_idx)), 'wb') as file:
            pickle.dump(result.best_model, file)
    fold_idx += 1

print('Grid search complete in: ', (datetime.now() - start_time))