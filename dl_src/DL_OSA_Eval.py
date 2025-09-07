import numpy as np
import pandas as pd
import warnings
import time

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
import torchvision.models as models
import torch
import torch.nn as nn
from easydict import EasyDict as edict
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import json
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
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
from sklearn.model_selection import StratifiedShuffleSplit
from datetime import datetime
import argparse
import itertools
from pathlib import Path
from models import MV_LSTM, MultivariateCNN
import os, pickle

device = 'cpu'

def parse_arguments():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model', type=str, help='Model type, current support lstm, cnn, rnn')
    parser.add_argument('--target_col', type=str, help='The column of the target label, choose Severity for multi-class, AHI_5 for binary cut-off at 5')
    parser.add_argument('--batch_size', type=int, help='batch_size', default=256)
    parser.add_argument('--display', type=int, help='batch_size', default=256)
    return parser

parser = parse_arguments()
args = parser.parse_args()

path = 'data/OSA_complete_patients.csv'

df= pd.read_csv(path, index_col = ['PatientID'])
df.drop(df.columns[[0]], axis=1, inplace=True)
df.head(5)

# I am going to add columns AHI5, AHI15, and AHI30 
df['AHI_5'] = df['Severity'].apply(lambda x: 1 if x >= 1 else 0)
df['AHI_15'] = df['Severity'].apply(lambda x: 1 if x >= 2 else 0)
df['AHI_30'] = df['Severity'].apply(lambda x: 1 if x >= 3 else 0)

print("AHI 5 value counts:", df['AHI_5'].value_counts(), "\n")
print("AHI 15 value counts:", df['AHI_15'].value_counts(), "\n")
print("AHI 30 value counts:", df['AHI_30'].value_counts(), "\n")

# Making sure this aligns with the severity column
print("Severity value counts:", df['Severity'].value_counts())
seq_length = 49
# Scaling
x = preprocessing.MinMaxScaler().fit_transform(df.values[:,:seq_length])
y = df[args.target_col].values

num_class = len(set(y))

print(np.unique(y))
model_path = 'results/{}/{}'.format(args.target_col ,args.model)
print(model_path)
Path(model_path).mkdir(parents=True, exist_ok=True)

def get_data(x_inp,y_inp):
    train_tensor = TensorDataset(torch.from_numpy(x_inp), torch.from_numpy(y_inp))
    train_loader = DataLoader(train_tensor, batch_size=args.batch_size, shuffle=False)

    return train_loader

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

#Train model
def eval_model(model, test_loader):
    model.eval()
    total_loss = 0.0
    out = []
    out_label = []
    for (inp, target) in test_loader:
        x_batch = inp.float().to(device).unsqueeze(dim=2)
        y_batch = target.long().to(device)
        if 'lstm' in args.model:
            model.init_hidden(x_batch.size(0))
        output = model(x_batch)
        #if epoch % 100 == 0:
        #    print(output.shape)
        loss = criterion(output, y_batch)
        total_loss += loss
        preds = F.log_softmax(output, dim=1).argmax(dim=1).detach().cpu().numpy()
        output_label = y_batch.detach().cpu().numpy()
        for val in preds:
            out.append(val)
        for val in output_label:
            out_label.append(val)
    eval_loss = total_loss / len(test_loader)
    return eval_loss, out, out_label


def evaluate(args, model_param, train_loader, test_loader, df_result, fold_num = 1):
    if args.model == 'lstm':
        model = MV_LSTM(device=device, n_features=1,seq_length=seq_length, hidden_dim=model_param[0], n_layers=model_param[2], num_class=num_class)
    if args.model == 'cnn':
        model = MultivariateCNN(dropout = model_param[0], num_class=num_class)
    best_model_path = os.path.join(model_path, 'best_model_fold_{}.pth'.format(fold_num))
    model.load_state_dict(torch.load(best_model_path))
    model.to(device)

    eval_loss, preds, test_label= eval_model(model, test_loader)
    train_loss, preds_train, train_label= eval_model(model, train_loader)

    row_test = eval_metric(test_label, preds, args, type ='test')
    row_train= eval_metric(train_label, preds_train, args, type ='train')
    df_result = pd.concat([df_result, pd.DataFrame([row_test])], ignore_index=True)
    df_result = pd.concat([df_result, pd.DataFrame([row_train])], ignore_index=True)

    return df_result


start_time = datetime.now()
criterion = nn.CrossEntropyLoss()
#sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
sss = KFold(n_splits=5, random_state=0, shuffle=True)
fold_idx = 1
df_result = pd.DataFrame(columns=['model', 'accuracy', 'recall_weighted', 'recall_macro', 'f1_weighted', 'f1_macro', 'bal_acc', 'precision_weighted', 'precision_macro', 'g_mean'])

for train_ids, test_ids in sss.split(x, y):
    print('Fold: ', fold_idx)
    train_loader, test_loader = get_data(x[train_ids], y[train_ids]), get_data(x[test_ids], y[test_ids])
    param_path = os.path.join(model_path, 'best_params_fold_{}.json'.format(fold_idx))
    with open(param_path) as f:
        model_param = json.load(f)
    df_result = evaluate(args, model_param, train_loader, test_loader, df_result, fold_num=fold_idx)
    fold_idx += 1

result_path = os.path.join(model_path,'full_metric_result.csv'.format(fold_idx))
df_result.to_csv(result_path, index = False)
print('Total time: ', (datetime.now() - start_time))