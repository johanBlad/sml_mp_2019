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

# Scatter plot for all parameters 
pd.plotting.scatter_matrix(test.iloc[:,0:13], figsize=(12,10), diagonal = 'kde') #mode and time_signature is almost same for all songs
plt.show()

trainCopy = train.copy()
X = trainCopy.drop(columns = ['duration', 'mode', 'time_signature'])
y = trainCopy.iloc[:,13]

x_test = test.iloc[:,0:13]
nFold = 10

#K = np.arange(1, 200)
methods = []
methods.append(skl_lm.LogisticRegression(penalty='l1', solver = 'liblinear', warm_start=True,))
methods.append(skl_da.LinearDiscriminantAnalysis())
methods.append(skl_da.QuadraticDiscriminantAnalysis())
methods.append(skl_nb.KNeighborsClassifier(2))

kf = skl_ms.KFold(n_splits = nFold, shuffle = True, random_state = 1) # n_splits: number of folds
# matrix: 10x4
missclassification = np.zeros((nFold, len(methods))) # needed several parantheses
#print(missclassification)

for i, (train_index, val_index) in enumerate(kf.split(X)):
    xTrain, xVal = X.iloc[train_index], X.iloc[val_index]
    yTrain, yVal = y.iloc[train_index], y.iloc[val_index]
    #print(xTrain)
    #for j, k in enumerate(K):
    for method in range(np.shape(methods)[0]):
        model = methods[method]
        model.fit(xTrain, yTrain)
        prediction = model.predict(xVal)
        missclassification[i, method] = np.mean(prediction != yVal) # if prediction != y -> missclassification

    #avergeError = np.mean(error, axis = 0) # why can we not initiate matrix?

#error /= nFold

plt.boxplot(missclassification)
plt.xticks(np.arange(4) + 1, ('LogReg', 'LDA', 'QDA', 'k-NN'))
plt.show()
