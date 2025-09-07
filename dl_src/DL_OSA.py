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

device =  'cuda' if torch.cuda.is_available() else 'cpu'

def parse_arguments():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model', type=str, help='Model type, current support lstm, cnn, rnn')
    parser.add_argument('--target_col', type=str, help='The column of the target label, choose Severity for multi-class, AHI_5 for binary cut-off at 5')
    parser.add_argument('--batch_size', type=int, help='batch_size', default=256)
    parser.add_argument('--epochs', type=int, help='batch_size', default=70)
    parser.add_argument('--display', type=int, help='batch_size', default=256)
    parser.add_argument('--imb', type=str, help='balance strategy', default=256)
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
Path(model_path).mkdir(parents=True, exist_ok=True)

def get_data(x_inp,y_inp):
    train_tensor = TensorDataset(torch.from_numpy(x_inp), torch.from_numpy(y_inp))
    train_loader = DataLoader(train_tensor, batch_size=args.batch_size, shuffle=False)

    return train_loader

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

def train_model(args, model, train_loader, test_loader, criterion, optimizer):
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        total_acc = 0.0
        for (inp, target) in train_loader:
            x_batch = inp.float()
            y_batch = target.long()
            x_batch = x_batch.to(device).unsqueeze(dim=2)
            y_batch = y_batch.to(device)
            if 'lstm' in args.model:
                model.init_hidden(x_batch.size(0))
            output = model(x_batch)
            #if epoch % 100 == 0:
            #    print(output.shape)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()        
            optimizer.zero_grad()
            total_loss += loss
            preds = F.log_softmax(output, dim=1).argmax(dim=1).detach().cpu().numpy()
            batch_acc = accuracy_score(preds, y_batch.detach().cpu().numpy())
            total_acc += batch_acc
        train_loss = total_loss / len(train_loader)
        train_acc = total_acc / len(train_loader)
        if epoch % args.display == 0:
            print('epoch : ' , epoch , '|train loss : ' , train_loss.item(), '|train acc : ' , train_acc)
    eval_loss, out, out_label = eval_model(model, test_loader)
    #out = np.squeeze(np.array(out), axis=1)
    #out_label = np.squeeze(np.array(out_label), axis=1)

    return model, out, out_label

def tune_LSTM(args, params, train_loader, test_loader, fold_num = 1):
    best_f1 = 0.0
    best_result = []
    best_model = []
    best_param = []
    for model_param in params:
        if args.model == 'lstm':
            model = MV_LSTM(device=device, n_features=1,seq_length=seq_length, hidden_dim=model_param[0], n_layers=model_param[2], num_class=num_class)
        if args.model == 'rnn':
            model = MV_RNN(device=device, n_features=1, seq_length=seq_length, hidden_dim=model_param[0], n_layers=model_param[2], num_class=num_class)
        if args.model == 'cnn':
            model = MultivariateCNN(dropout = model_param[0], num_class=num_class)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr= model_param[1])

        model, out, out_label= train_model(args, model, train_loader, test_loader, criterion, optimizer)
        print(np.unique(out))
        print(np.unique(out_label))
        result = {
            'accuracy': accuracy_score(out_label, out),
            'recall': recall_score(out_label, out, average='weighted'),
            'f1_weighted': f1_score(out_label, out, average='weighted'),
            'f1_macro': f1_score(out_label, out, average='macro'),
            'bal_acc': balanced_accuracy_score(out_label, out),
            'precision': precision_score(out_label, out, average='weighted'),
            'g_mean': geometric_mean_score(out_label, out)
            }

        if result['f1_macro'] > best_f1:
            best_f1 = result['f1_macro']
            best_model = model
            best_result = result
            best_param = model_param

    torch.save(best_model.state_dict(), os.path.join(model_path, 'best_model_fold_{}.pth'.format(fold_num)))

    with open(os.path.join(model_path, 'best_params_fold_{}.json'.format(fold_num)), 'w') as f:
        json.dump(best_param, f)
    # Now, let's save this model to a file
    with open(os.path.join(model_path, 'best_model_fold_{}.pkl'.format(fold_num)), 'wb') as file:
        pickle.dump(best_model, file)

with open('config/{}.json'.format(args.model), "r") as f:
    search_space = json.load(f)

all_perm = []
for k, v in search_space.items():
    all_perm.append(v)
print(all_perm)

all_params = list(itertools.product(*all_perm))

start_time = datetime.now()
criterion = nn.CrossEntropyLoss()
#sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
sss = KFold(n_splits=5, random_state=0, shuffle=True)
fold_idx = 1
for train_ids, test_ids in sss.split(x, y):
    print('Fold: ', fold_idx)
    train_loader, test_loader = get_data(x[train_ids], y[train_ids]), get_data(x[test_ids], y[test_ids])
    tune_LSTM(args, all_params, train_loader, test_loader, fold_num=fold_idx)
    fold_idx += 1
print('Total time: ', (datetime.now() - start_time))
