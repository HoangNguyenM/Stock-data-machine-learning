import pandas as pd

from sklearn.ensemble import RandomForestClassifier

# get train and test data
train_data = pd.read_csv('D:/Dropbox/Code/traindata-tsla.csv')

train_labels = pd.read_csv('D:/Dropbox/Code/trainlabels-tsla.csv')

test_data = pd.read_csv('D:/Dropbox/Code/traindata-aapl.csv')

test_labels = pd.read_csv('D:/Dropbox/Code/trainlabels-aapl.csv')

# format train and test data
train_data.drop(train_data.columns[0], axis = 1, inplace=True)
train_labels.drop(train_labels.columns[0], axis = 1, inplace=True)
test_data.drop(test_data.columns[0], axis = 1, inplace=True)
test_labels.drop(test_labels.columns[0], axis = 1, inplace=True)

train_labels = train_labels.values.reshape(len(train_labels), 1, order='F')

test_labels = test_labels.values.reshape(len(test_labels), 1, order='F')

# run random forest
clf = RandomForestClassifier(n_estimators=20, max_depth=len(train_data.columns), random_state=0)
model = clf.fit(train_data, train_labels.ravel())

test_predict = model.predict(test_data)

sum = 0
for i in range(0, len(test_predict)):
    if (test_predict[i] - test_labels[i] == 0):
        sum += 1
        
print(sum/len(test_labels))
