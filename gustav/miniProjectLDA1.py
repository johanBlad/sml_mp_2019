import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sklearn.preprocessing as skl_pre
import sklearn.linear_model as skl_lm
import sklearn.discriminant_analysis as skl_da
import sklearn.neighbors as skl_nb
import sklearn.model_selection as skl_ms


# 200 songs x 13 features
test = pd.read_csv('songs_to_classify.csv')
# 750 songs x 13 features
train = pd.read_csv('training_data.csv')

yAll = train.iloc[:,13]
XAll = train.iloc[:,0:13]

trainCopy = train.copy()
indexNames = trainCopy[trainCopy['speechiness'] > 0.66].index
trainCopy.drop(indexNames, inplace = True)
#print(trainCopy)

X = trainCopy.drop(columns = ['duration', 'instrumentalness', 'speechiness'])
y = trainCopy.iloc[:,13]
print(y)
print('above is y')
x_test = test.iloc[:,0:13]

#X = train.drop(columns = ['duration'])
#print(X)

nFold = 10
kf = skl_ms.KFold(n_splits = nFold, shuffle = True, random_state = 2) # n_splits: number of folds
missclassification = np.zeros(nFold)
accuracy = np.zeros(nFold)
scaler = skl_pre.StandardScaler().fit(X)

for train_index, test_index in kf.split(X):
    xTrain, xTest = X.iloc[train_index], X.iloc[test_index]
    yTrain, yTest = y.iloc[train_index], y.iloc[test_index]
    #print(xTrain)
    for j in range(nFold):
        model = skl_da.LinearDiscriminantAnalysis()
        model.fit(scaler.transform(xTrain), yTrain)
        prediction = model.predict(scaler.transform(xTest))
        missclassification[j] = np.mean(prediction != yTest) # if prediction != y -> missclassification
        accuracy[j] = np.mean(prediction == yTest)

#avergeMiss = np.mean(missclassification, axis = 0) # why can we not initiate matrix?
#missclassification /= nFold
#accuracy /= nFold
print('Max Accuracy')
print(np.max(accuracy))
print('Min missclassification')
print(np.min(missclassification))
#print(missclassification[np.min(missclassification)])
#plt.plot(K, missclassification) # best for k between 5 and 60
#plt.show()


scaler = skl_pre.StandardScaler().fit(XAll)

model2 = skl_da.LinearDiscriminantAnalysis()
model2.fit(scaler.transform(XAll), yAll)

prediction = model2.predict(scaler.transform(XAll))
predictionTest = model2.predict(scaler.transform(x_test))

#prediction = np.empty(len(xTest), dtype = object)
# större än 0.5 -> 1
# mindre än 0.5 -> 0
#prediction = np.where(prediction[:, 0] >= 0.5, '1', '0')
print(predictionTest)

error = np.mean(prediction != yAll)
accuracy = np.mean(prediction == yAll)
print('Error:')
print(error)
print('Accuracy')
print(accuracy)



# xTrain, xVal, yTrain, yVal = skl_ms.train_test_split(X, y, test_size = 0.3, random_state = 1)
#
# K = np.arange(1, 200)
# missclassification = np.zeros((10, len(K))) # matrix: 10x200
# scaler = skl_pre.StandardScaler().fit(xTrain)
# #avergeMiss = []
# #test_size = np.arange(0.1, 0.9)
#
# for i in range(10):
#     xTrain, xVal, yTrain, yVal = skl_ms.train_test_split(X, y, test_size = 0.3)
#
#     for j, k in enumerate(K):
#         modelKNN = skl_nb.KNeighborsClassifier(k)
#         modelKNN.fit(scaler.transform(xTrain), yTrain)
#         yProbTestKNN = modelKNN.predict(scaler.transform(xVal))
#
#         missclassification[i, j] = np.mean(yProbTestKNN != yVal) # if prediction != y -> missclassification
#
# avergeMiss = np.mean(missclassification, axis = 0) # why can we not initiate matrix?
# #print(avergeMiss)
# minMiss = np.min(avergeMiss)
# print(minMiss)
# print(int(minMiss))
# print(K[int(minMiss)])
# #print(K.iloc[minMiss])
# print(K[np.where(int(minMiss))])
# #print(K)
# K = np.linspace(1, 199, 199)
#
# plt.plot(K, avergeMiss) # best for k = 25
# plt.show()
