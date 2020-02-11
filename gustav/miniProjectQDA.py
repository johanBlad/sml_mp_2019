
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sklearn.preprocessing as skl_pre
import sklearn.linear_model as skl_lm
import sklearn.discriminant_analysis as skl_da
import sklearn.neighbors as skl_nb

#from IPython.display import set_matplotlib_formats
#set_matplotlib_formats('png')
from IPython.core.pylabtools import figsize
figsize(10, 6) # Width and hight
plt.style.use('seaborn-white')



train = pd.read_csv('training_data.csv')
toClassify = pd.reac_csv('songs_to_classify.csv')
desc = train.describe()
#print(desc)
#pd.plotting.scatter_matrix(train.iloc[:,1:13],figsize=(8,8))
#plt.show()


np.random.seed(1)
trainI = np.random.choice(train.shape[0], size=750, replace=False)
trainIndex = train.index.isin(trainI)
train = train.iloc[trainIndex]
test = train.iloc[:,0:14]
#test.describe()


model = skl_da.QuadraticDiscriminantAnalysis()
X_train = train[['danceability', 'instrumentalness', 'energy', 'liveness']]
Y_train = train['label']
X_test = test[['danceability', 'instrumentalness', 'energy', 'liveness']]
Y_test = test['label']
model.fit(X_train, Y_train)
#print('Model summary:')
#print(model)
predict_prob = model.predict_proba(X_test)
#print('The class order in the model:')
#print(model.classes_)
print('Examples of predicted probabilities for the above classes:')
with np.printoptions(suppress=True, precision=3):
    print(predict_prob[0:5])  #inspect the first thirty predictions

prediction = np.empty(len(X_test), dtype=object)
prediction = np.where(predict_prob[:, 0]>=0.5, 0, 1)
print(prediction[0:30])

# Confusion matrix
print("Confusion matrix:\n")
print(pd.crosstab(prediction, Y_test), '\n')
# Accuracy
print(f"Accuracy: {np.mean(prediction == Y_test):.3f}")