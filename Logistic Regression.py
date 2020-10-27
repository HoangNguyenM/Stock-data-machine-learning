import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

# get train and test data

train_data = pd.read_csv('D:/Dropbox/Code/traindata-tsla.csv')

train_labels = pd.read_csv('D:/Dropbox/Code/trainlabels-tsla.csv')

test_data = pd.read_csv('D:/Dropbox/Code/traindata-aapl.csv')

test_labels = pd.read_csv('D:/Dropbox/Code/trainlabels-aapl.csv')

train_data.drop(train_data.columns[0], axis = 1, inplace=True)
train_labels.drop(train_labels.columns[0], axis = 1, inplace=True)
test_data.drop(test_data.columns[0], axis = 1, inplace=True)
test_labels.drop(test_labels.columns[0], axis = 1, inplace=True)

train_labels = train_labels.values.reshape(len(train_labels), 1, order='F')

test_labels = test_labels.values.reshape(len(test_labels), 1, order='F')

input_dim = len(train_data.columns)

train_data = torch.tensor(train_data.to_numpy(dtype = 'float32'))

train_labels = torch.tensor(train_labels)

test_data = torch.tensor(test_data.to_numpy(dtype = 'float32'))

test_labels = torch.tensor(test_labels)

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs

n_iters = 2
lr_rate = 0.001

model = LogisticRegression(input_dim, 1)
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(),lr = 0.00001)


for  iter in range(0, n_iters):
    optimizer.zero_grad()
    output = model.forward(train_data)
    loss = criterion(output,train_labels)
    loss.backward()
    optimizer.step()


test_predict = model.forward(test_data)

sum = 0
for i in range(0, len(test_predict)):
    if (test_predict[i] - test_labels[i] == 0):
        sum += 1
    print(test_predict[i])
        
print(sum/len(test_labels))
