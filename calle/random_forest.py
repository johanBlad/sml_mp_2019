from utility import initialize_data, predict
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier #BaggingClassifier ???
import sklearn.model_selection as skl_ms


data_dict = initialize_data()

depth = 2
lr = 0.017
n_est = 100
n_fold = 10

accuracy = np.zeros(n_fold)
x_train = data_dict.get('x_train')
y_train = data_dict.get('y_train')
x_test = data_dict.get('x_test')

cv = skl_ms.KFold(n_fold, shuffle=True, random_state=1)
model = RandomForestClassifier(n_estimators = n_est) #.fit(x_train, y_train)

for i, (train_index, val_index) in enumerate(cv.split(x_train)):
    x_train_l, x_val = x_train[train_index], x_train[val_index]
    y_train_l, y_val = y_train[train_index], y_train[val_index]

    model.fit(x_train_l, y_train_l)
    prediciton = model.predict(x_val)
    accuracy[i] = 1 - np.mean(prediciton != y_val)
model.fit(x_train_l, y_train_l)
avg_acc = np.mean(accuracy)
print('Accuracy:', accuracy)
print('Average: ', avg_acc, '\n')
print(prediciton)