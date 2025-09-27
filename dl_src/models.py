import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

from sklearn.base import BaseEstimator

from torch.nn import Sequential, LSTM, GRU, RNN, Conv1d, MaxPool1d, BatchNorm1d
from torch.utils.data import DataLoader, TensorDataset


class SelectItem(nn.Module):
    def __init__(self, item_index):
        super(SelectItem, self).__init__()
        self.item_index = item_index

    def forward(self, inputs):
        
        # print('Hidden shape:', inputs[-1].shape)
        # exit()
        if len(self.item_index) == 1:
            return inputs[self.item_index[0]]
        
        if len(self.item_index) == 2:
            return inputs[self.item_index[0]][self.item_index[1]]
        
        if len(self.item_index) == 3:
            return inputs[self.item_index[0]][self.item_index[1]][self.item_index[2]]
        
        if len(self.item_index) == 4:
            return inputs[self.item_index[0]][self.item_index[1]][self.item_index[2]][self.item_index[3]]
    

class LSTMCustom(BaseEstimator):
    def __init__(self, input_dim, hidden_dim, n_layers, n_classes, n_epochs, lr=0.001):
        self.n_classes = n_classes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_epochs = n_epochs
        self.lr = lr
        self.model = None

    def fit(self, x_train, y_train):
        self.model = Sequential(LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=self.n_layers, batch_first=True),
                                SelectItem((-1, 0, -1)),
                                nn.Linear(self.hidden_dim, self.n_classes))
        self.model = self.model.to('cuda')

        dataset = TensorDataset(torch.tensor(x_train).float(), torch.tensor(y_train).long())
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for epoch in range(self.n_epochs):  # number of epochs
            tqdm_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.n_epochs}")
            for inputs, labels in tqdm_bar:
                inputs = inputs.unsqueeze(-1)  # Add feature dimension
                inputs, labels = inputs.to('cuda'), labels.to('cuda')

                optimizer.zero_grad()
                outputs = self.model(inputs)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                tqdm_bar.set_postfix(loss=loss.item())


    def predict(self, x_test):
        self.model.eval()
        with torch.no_grad():
            inputs = torch.tensor(x_test).float()
            inputs = inputs.unsqueeze(-1)  # Add feature dimension
            inputs = inputs.to('cuda')
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs.data, 1)

        return predicted.cpu().numpy()


class GRUCustom(BaseEstimator):
    def __init__(self, input_dim, hidden_dim, n_layers, n_classes, n_epochs, lr=0.001):
        self.n_classes = n_classes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_epochs = n_epochs
        self.lr = lr
        self.model = None
        
    def fit(self, x_train, y_train):
        self.model = Sequential(GRU(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=self.n_layers, batch_first=True),
                                SelectItem((-1, -1)),
                                nn.Linear(self.hidden_dim, self.n_classes))
        self.model = self.model.to('cuda')

        dataset = TensorDataset(torch.tensor(x_train).float(), torch.tensor(y_train).long())
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for epoch in range(self.n_epochs):  # number of epochs
            tqdm_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.n_epochs}")
            for inputs, labels in tqdm_bar:
                inputs = inputs.unsqueeze(-1)  # Add feature dimension
                inputs, labels = inputs.to('cuda'), labels.to('cuda')

                optimizer.zero_grad()
                outputs = self.model(inputs)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                tqdm_bar.set_postfix(loss=loss.item())


    def predict(self, x_test):
        self.model.eval()
        with torch.no_grad():
            inputs = torch.tensor(x_test).float()
            inputs = inputs.unsqueeze(-1)  # Add feature dimension
            inputs = inputs.to('cuda')
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs.data, 1)

        return predicted.cpu().numpy()


class RNNCustom(BaseEstimator):
    def __init__(self, input_dim, hidden_dim, n_layers, n_classes, n_epochs, lr=0.001):
        self.n_classes = n_classes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_epochs = n_epochs
        self.lr = lr
        self.model = None

    def fit(self, x_train, y_train):
        self.model = Sequential(RNN(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=self.n_layers, batch_first=True),
                                SelectItem((-1, -1)),
                                nn.Linear(self.hidden_dim, self.n_classes))
        self.model = self.model.to('cuda')

        dataset = TensorDataset(torch.tensor(x_train).float(), torch.tensor(y_train).long())
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for epoch in range(self.n_epochs):  # number of epochs
            tqdm_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.n_epochs}")
            for inputs, labels in tqdm_bar:
                inputs = inputs.unsqueeze(-1)  # Add feature dimension
                inputs, labels = inputs.to('cuda'), labels.to('cuda')

                optimizer.zero_grad()
                outputs = self.model(inputs)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                tqdm_bar.set_postfix(loss=loss.item())


    def predict(self, x_test):
        self.model.eval()
        with torch.no_grad():
            inputs = torch.tensor(x_test).float()
            inputs = inputs.unsqueeze(-1)  # Add feature dimension
            inputs = inputs.to('cuda')
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs.data, 1)

        return predicted.cpu().numpy()    


class CNNCustom(BaseEstimator):
    def __init__(self, n_features, kernel_size, n_layers, n_classes, n_epochs, lr=0.001):
        self.n_features = n_features
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.n_epochs = n_epochs
        self.lr = lr
        self.model = None

    def fit(self, x_train, y_train):
        layers = []
        n_channels = 1
        data_shape = self.n_features
        for _ in range(self.n_layers):
            layers.append(Conv1d(in_channels=n_channels, out_channels=n_channels * 2, kernel_size=self.kernel_size, padding=1))
            layers.append(MaxPool1d(kernel_size=2, stride=2))
            layers.append(BatchNorm1d(n_channels * 2))
            layers.append(nn.ReLU())

            n_channels *= 2
            data_shape = data_shape // 2

        layers.append(nn.Flatten())
        layers.append(nn.Linear(data_shape * n_channels, self.n_classes))

        self.model = nn.Sequential(*layers)
        self.model.to('cuda')

        dataset = TensorDataset(torch.tensor(x_train).float(), torch.tensor(y_train).long())
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for epoch in range(self.n_epochs):  # number of epochs
            tqdm_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.n_epochs}")
            for inputs, labels in tqdm_bar:
                inputs = inputs.unsqueeze(1)  # Add feature dimension
                inputs, labels = inputs.to('cuda'), labels.to('cuda')

                optimizer.zero_grad()
                outputs = self.model(inputs)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                tqdm_bar.set_postfix(loss=loss.item())


    def predict(self, x_test):
        self.model.eval()
        with torch.no_grad():
            inputs = torch.tensor(x_test).float()
            inputs = inputs.unsqueeze(1)  # Add feature dimension
            inputs = inputs.to('cuda')
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs.data, 1)

        return predicted.cpu().numpy()


############# GCN ############
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, adj, use_bias=True):
        super(GCNLayer, self).__init__()
        self.adj = adj
        self.weight = nn.Parameter(torch.FloatTensor(torch.zeros(size=(in_features, out_features))))
        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(torch.zeros(size=(out_features,))))
        else:
            self.register_parameter('bias', None)

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        # print('X shape: ', x.shape)
        # print('weight shape: ', self.weight.shape)
        x = x @ self.weight
        if self.bias is not None:
            x += self.bias

        # print('Adj shape: ', self.adj.shape)
        # print('x shape: ', x.shape)
        return torch.matmul(self.adj, x)

class GCNCustom(BaseEstimator):
    def __init__(self, n_features, in_features, hidden_dim, n_layers, n_classes, n_epochs, lr=0.001):
        self.n_features = n_features
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.n_epochs = n_epochs
        self.lr = lr

    def fit(self, x_train, y_train):
        adj = np.random.randint(0, 2, size=(self.n_features, self.n_features))
        for i in range(len(adj)):
            adj[i][i] = 1
        adj = adj.astype(np.float32)
        adj = torch.from_numpy(adj).to('cuda')

        layers = []
        layers.append(GCNLayer(self.in_features, self.hidden_dim, adj, use_bias=True))
        layers.append(nn.ReLU())
        for _ in range(self.n_layers - 1):
            layers.append(GCNLayer(self.hidden_dim, self.hidden_dim, adj, use_bias=True))
            layers.append(nn.ReLU())

        layers.append(nn.Flatten())
        layers.append(nn.Linear(self.n_features * self.hidden_dim, self.n_classes))

        self.model = nn.Sequential(*layers)
        self.model.to('cuda')

        dataset = TensorDataset(torch.tensor(x_train).float(), torch.tensor(y_train).long())
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for epoch in range(self.n_epochs):  # number of epochs
            tqdm_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.n_epochs}")
            for inputs, labels in tqdm_bar:
                inputs = inputs.unsqueeze(2)  # Add feature dimension
                inputs, labels = inputs.to('cuda'), labels.to('cuda')

                optimizer.zero_grad()
                outputs = self.model(inputs)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                tqdm_bar.set_postfix(loss=loss.item())


    def predict(self, x_test):
        self.model.eval()
        with torch.no_grad():
            inputs = torch.tensor(x_test).float()
            inputs = inputs.unsqueeze(2)  # Add feature dimension
            inputs = inputs.to('cuda')
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs.data, 1)

        return predicted.cpu().numpy()
    

if __name__ == "__main__":
    x_train = np.random.rand(100, 27)
    y_train = np.random.randint(0, 2, size=(100,))
    x_test = np.random.rand(20, 27)

    model = GCNCustom(n_features=27, in_features=1, hidden_dim=8, n_layers=2, n_classes=2, n_epochs=10, lr=0.001)
    # from torchsummary import summary
    # summary(model, x_train.shape, batch_size=64)

    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    print("CNN Predictions:", predictions)

