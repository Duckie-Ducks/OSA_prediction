import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class MV_LSTM(torch.nn.Module):
    def __init__(self,device, n_features, seq_length, hidden_dim = 100, n_layers = 2, num_class=2):
        super(MV_LSTM, self).__init__()
        self.n_features = n_features
        self.seq_len = seq_length
        self.device=device
        self.n_hidden = hidden_dim # number of hidden states
        self.n_layers = n_layers # number of LSTM layers (stacked)
        
        self.l_lstm = torch.nn.LSTM(input_size = n_features, 
                                 hidden_size = self.n_hidden,
                                 num_layers = self.n_layers, 
                                 batch_first = True)
        '''
        if model_type == 'rnn':
            self.l_lstm = torch.nn.RNN(input_size = n_features, 
                                     hidden_size = self.n_hidden,
                                     num_layers = self.n_layers, 
                                     batch_first = True)
        '''
        # according to pytorch docs LSTM output is 
        # (batch_size,seq_len, num_directions * hidden_size)
        # when considering batch_first = True
        self.l_linear = torch.nn.Linear(self.n_hidden*self.seq_len, num_class)
        
    
    def init_hidden(self, batch_size):
        # even with batch_first = True this remains same as docs
        hidden_state = torch.zeros(self.n_layers,batch_size,self.n_hidden, device=self.device)
        cell_state = torch.zeros(self.n_layers,batch_size,self.n_hidden, device=self.device)
        self.hidden = (hidden_state, cell_state)
    
    
    def forward(self, x):        
        batch_size, seq_len, _ = x.size()
        lstm_out, self.hidden = self.l_lstm(x,self.hidden)
        # lstm_out(with batch_first = True) is 
        # (batch_size,seq_len,num_directions * hidden_size)
        # for following linear layer we want to keep batch_size dimension and merge rest       
        # .contiguous() -> solves tensor compatibility error
        x = lstm_out.contiguous().view(batch_size,-1)
        return self.l_linear(x)

# Define Pytorch CNN model
class MultivariateCNN(nn.Module):
    def __init__(self, num_filters=[512, 1024, 2048], dropout=0.01, n_features = 1, num_class = 2):
        super(MultivariateCNN, self).__init__()
        self.filter_sizes = [1, 2, 3]
        self.conv1d_list = nn.ModuleList(
            [nn.Conv1d(n_features, num_filters[i], self.filter_sizes[i]) for i in range(len(self.filter_sizes))])
        self.linear = nn.Linear(np.sum(num_filters), num_class)
        self.dropout = nn.Dropout(dropout)
       

    def forward(self, X_batch):
        X_batch = X_batch.permute(0, 2, 1)  # (batch_size, n_features, n_look_back)
        X_conv_list = [F.relu(conv1d(X_batch)) for conv1d in self.conv1d_list]
        X_pool_list = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2]) for x_conv in X_conv_list]
        X_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in X_pool_list], dim=1)
        y_predicted = self.linear(self.dropout(X_fc))

        return y_predicted

