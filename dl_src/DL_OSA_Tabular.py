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

def parse_arguments():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model', type=str, help='Model type, current support GANDALF, FTT, TabNet, TabTransformer, CEM, GATE, Node, AutoInt, DANet')
    parser.add_argument('--target_col', type=str, help='The column of the target label, choose Severity for multi-class, AHI_5 for binary cut-off at 5')
    parser.add_argument('--batch_size', type=int, help='batch_size', default=256)
    return parser

def F1(y_true, y_pred):
    return f1_score(y_true, y_pred['prediction'], average='macro')

def split_data(data, portion = 0.8):
    train_idx = int(len(data) * 0.8)

    return data[:train_idx], data[train_idx:]

def eval_metric(test, preds, args, type):
    # Calculate metrics and update DataFrame
    metrics = {
        'accuracy': accuracy_score(test, preds),
        'recall': recall_score(test, preds, average='weighted'),
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

def load_OSA(target_col):
    path = 'data/OSA_complete_patients.csv'

    df= pd.read_csv(path, index_col = ['PatientID'])
    df.drop(df.columns[[0]], axis=1, inplace=True)
    df.head(5)
   
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
    return df, cat_cols, cont_cols, target_col

#data, cat_col_names, num_col_names, target_col = load_covertype_dataset()
start_time = datetime.now()
parser = parse_arguments()
args = parser.parse_args()
print(args)
data, cat_col_names, num_col_names, target_col = load_OSA(target_col=args.target_col)

print('cat_col_names: ', cat_col_names)
print('num_col_names: ', num_col_names)

model_configs = {'GANDALF':GANDALFConfig(task="classification"),
                'FTT':FTTransformerConfig(task="classification"), 
                'TabNet':TabNetModelConfig(task="classification"), 
                'TabTransformer':TabTransformerConfig(task="classification"),
                'CEM': CategoryEmbeddingModelConfig(task="classification"), 
                'GATE':GatedAdditiveTreeEnsembleConfig(task="classification"), 
                'Node':NodeConfig(task="classification"), 
                'DANet':DANetConfig(task="classification"), 
                'AutoInt':AutoIntConfig(task="classification")}


model_config = model_configs[args.model]

sss = KFold(n_splits=5, random_state=0, shuffle=True)

model_path = 'results/{}/{}'.format(args.target_col ,args.model)
Path(model_path).mkdir(parents=True, exist_ok=True)

train, test = split_data(data, portion = 0.8)
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

tabular_model = TabularModel(
    data_config=data_config,
    model_config=model_config,
    optimizer_config=optimizer_config,
    trainer_config=trainer_config,
)
tabular_model.fit(train=train, validation=test)
result = tabular_model.evaluate(test)
pred_df = tabular_model.predict(test)
preds = pred_df['prediction']
score = eval_metric(test[target_col], preds, args, type = 'Test')

print('Score: ', score)
print('Complete in: ', (datetime.now() - start_time))