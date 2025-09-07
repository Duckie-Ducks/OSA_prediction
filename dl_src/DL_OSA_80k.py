from pytorch_tabular.models import (
    GANDALFConfig, 
    FTTransformerConfig, 
    TabNetModelConfig, 
    TabTransformerConfig, 
    CategoryEmbeddingModelConfig, 
    GatedAdditiveTreeEnsembleConfig,
    NodeConfig, #Computational extensive, batch size should be smaller for smaller GPU
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
from sklearn.model_selection import KFold
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
from datetime import datetime
import pickle
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

def parse_arguments():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model', type=str, help='Model type, current support GANDALF, FTT, TabNet, TabTransformer, CEM, GATE, Node, AutoInt, DANet')
    parser.add_argument('--target_col', type=str, help='The column of the target label, choose Severity for multi-class, AHI_5 for binary cut-off at 5')
    parser.add_argument('--batch_size', type=int, help='batch_size', default=256)
    parser.add_argument('--imb', type=str, help='Imbalance handling strategy, choose ADASYN or SMOTE', default=256)
    parser.add_argument('--imp', type=str, help='Imp handling strategy, choose ADASYN or SMOTE', default=256)
    parser.add_argument('--uncleaned_data', type=int, help='Imp handling strategy, 1: Mean/0, 2:Mean/Most Freq, 3:Median/0, 4:Median/Most Freq', default=0)
    parser.add_argument('--features_name', type=str, help="Choose collection of features for learning", choices=["all_features", "Demographic", "Measurements", "Symptoms", "Questionnaires", 'Comorbidities', "Mencar", "Wu", "Huang", "Rodruiges", "Ustun"])
    return parser

def F1(y_true, y_pred):
    return f1_score(y_true, y_pred['prediction'], average='macro')

def split_data(data, portion = 0.8):
    train_idx = int(len(data) * 0.8)

    return data[:train_idx], data[train_idx:]

def Imb_Transform(train, imb_func):
    Data_Col_Lim = train.shape[1] - 4
    X_train, y_train = train.values[:,:Data_Col_Lim], train[args.target_col]
    X_train_bal, y_train_bal = imb_func.fit_resample(X_train, y_train)
    list_header = list(train)[:Data_Col_Lim] + [args.target_col]
    train_bal = pd.DataFrame(columns=list_header)
    for i in range(Data_Col_Lim):
        header = list_header[i]
        train_bal[header] = X_train_bal[:,i]
    train_bal[args.target_col] = y_train_bal

    return train_bal

def get_feature(feature_name):
    with open("config/Feature_Selection/feature_config.json") as f:
        feature_collections = json.load(f)

    return feature_collections[feature_name]

def eval_metric(test, preds, args, type):
    # Calculate metrics and update DataFrame
    metrics = {
        'accuracy': accuracy_score(test, preds),
        'recall': recall_score(test, preds, average='macro'),
        'f1': f1_score(test, preds, average='weighted'),
        'bal_acc': balanced_accuracy_score(test, preds),
        'precision': precision_score(test, preds, average='weighted'),
        'g_mean': geometric_mean_score(test, preds)
    }

    row = {'model': args.model + str(type),
           'accuracy': round(metrics['accuracy'], 3),
           'recall': round(metrics['recall'], 3),
           'f1': round(metrics['f1'], 3),
           'bal_acc': round(metrics['bal_acc'], 3),
           'precision': round(metrics['precision'], 3),
           'g_mean': round(metrics['g_mean'], 3)}

    return row

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


def load_OSA(args):
    if args.uncleaned_data:
        path = 'data/OSA_classifier.csv'
    else:
        path = 'data/OSA_complete_patients.csv'

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
    cat_cols = [i for i in features if i not in continuous]

    mean0 = df.copy(deep= True)           # copy original df

    if args.imp == 1:
        imp_cont = SimpleImputer(strategy='mean', missing_values=np.nan)  # define imputer for continuous variables
        imp_cat = SimpleImputer(strategy='constant', missing_values=np.nan, fill_value = 0)  # define imputer for categorical variables
    elif args.imp == 2:
        imp_cont = SimpleImputer(strategy='mean', missing_values=np.nan)  # define imputer for continuous variables
        imp_cat = SimpleImputer(strategy='most_frequent', missing_values=np.nan, fill_value = 0)  # define imputer for categorical variables
    elif args.imp == 3:
        imp_cont = SimpleImputer(strategy='median', missing_values=np.nan)  # define imputer for continuous variables
        imp_cat = SimpleImputer(strategy='constant', missing_values=np.nan, fill_value = 0)  # define imputer for categorical variables
    elif args.imp == 4:
        imp_cont = SimpleImputer(strategy='median', missing_values=np.nan)  # define imputer for continuous variables
        imp_cat = SimpleImputer(strategy='most_frequent', missing_values=np.nan, fill_value = 0)  # define imputer for categorical variables
    

    if args.imp != 0:
        imp_cont = imp_cont.fit(mean0[continuous])                    # fit imputer on columns
        mean0[continuous] = imp_cont.transform(mean0[cont_cols])     # transform columns
        imp_cat = imp_cat.fit(mean0[categorical])                                  # fit imputer on columns
        mean0[categorical] = imp_cat.transform(mean0[cat_cols])
        df = mean0

    # I am going to add columns AHI5, AHI15, and AHI30 
    df['AHI_5'] = df['Severity'].apply(lambda x: 1 if x >= 1 else 0)
    df['AHI_15'] = df['Severity'].apply(lambda x: 1 if x >= 2 else 0)
    df['AHI_30'] = df['Severity'].apply(lambda x: 1 if x >= 3 else 0)
    return df, cat_cols, cont_cols, args.target_col


#data, cat_col_names, num_col_names, target_col = load_covertype_dataset()
start_time = datetime.now()
parser = parse_arguments()
args = parser.parse_args()
print(args)
data, cat_col_names, num_col_names, target_col = load_OSA(args)

print('cat_col_names: ', cat_col_names)
print('num_col_names: ', num_col_names)

model_configs = {'GANDALF':GANDALFConfig,
                'FTT':FTTransformerConfig, 
                'TabNet':TabNetModelConfig, 
                'TabTransformer':TabTransformerConfig,
                'CEM': CategoryEmbeddingModelConfig, 
                'GATE':GatedAdditiveTreeEnsembleConfig, 
                'Node':NodeConfig, 
                'DANet':DANetConfig, 
                'AutoInt':AutoIntConfig}


model_config = model_configs[args.model]

sss = KFold(n_splits=5, random_state=0, shuffle=True)

with open('config/{}.json'.format(args.model), "r") as f:
    search_space = json.load(f)

model_path = 'results_all_feature/{}/{}'.format(args.target_col ,args.model)
Path(model_path).mkdir(parents=True, exist_ok=True)

imb_path = 'results_imb_imp/{}/{}/{}/{}/{}'.format(args.target_col, args.imb, args.imp, args.features_name,args.model)
Path(imb_path).mkdir(parents=True, exist_ok=True)

adasyn = ADASYN(random_state=0)
smote = SMOTE(random_state=0)
borderline = BorderlineSMOTE(random_state=0)
under = RandomUnderSampler(random_state=0)
from sklearn import metrics
import pandas as pd

fold_idx = 1
for train_ids, test_ids in sss.split(data):
    train = data.iloc[train_ids]
    test =  data.iloc[test_ids]
    if args.imb == 'ADASYN':
        train_bal = Imb_Transform(train, adasyn)
    elif args.imb == 'SMOTE':
        train_bal = Imb_Transform(train, smote)
    elif args.imb == 'Borderline':
        train_bal = Imb_Transform(train, borderline)
    elif args.imb == 'Under':
        train_bal = Imb_Transform(train, under)
    else: #No balance
        train_bal = train
    param_path = os.path.join(model_path, 'best_params_fold_{}.json'.format(fold_idx))
    with open(param_path) as f:
        model_param = json.load(f)
    specific_config = model_config(task="classification", learning_rate = model_param['optimizer_config__learning_rate'])
    
    data_config = DataConfig(
        target=[
            target_col
        ],  # target should always be a list. Multi-targets are only supported for regression. Multi-Task Classification is not implemented
        continuous_cols=num_col_names,
        categorical_cols=cat_col_names,
    )
    trainer_config = TrainerConfig(
        batch_size=args.batch_size,
        checkpoints_every_n_epochs = 150,
        load_best = False,
        max_epochs = int(model_param['trainer_config__max_epochs'])
    )

    optimizer_config = OptimizerConfig()
    
    tabular_model = TabularModel(
        data_config=data_config,
        model_config=specific_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
    )
    tabular_model.fit(train=train_bal, validation=test)
    result = tabular_model.evaluate(test)
    pred_df = tabular_model.predict(test)
    preds = pred_df['prediction']
    score = eval_metric(test[target_col], preds, args, type = 'Test')
    print('Score: ', score)
    with open(os.path.join(imb_path, 'best_model_fold_{}.pkl'.format(fold_idx)), 'wb') as file:
            pickle.dump(tabular_model, file)
    fold_idx += 1
    #Confusion Matrix
    confusion_matrix = metrics.confusion_matrix(test[args.target_col], preds)
    print(confusion_matrix)
    y_true = pd.Series(test[args.target_col], name="Actual")
    y_pred = pd.Series(preds, name="Predicted")
    df_confusion = pd.crosstab(y_true, y_pred)
    result_path = os.path.join(model_path,'confusion_matrix.csv'.format(fold_idx))
    df_confusion.to_csv(result_path)
    #exit()

print('Complete in: ', (datetime.now() - start_time))