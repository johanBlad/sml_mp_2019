from utility import initialize_data, predict
import sklearn.linear_model as skl_lm
import sklearn.model_selection as mod_sel
import sklearn.tree as sk_tree
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
import os
import sklearn.model_selection as skl_ms

data_dict = initialize_data()

x_train = data_dict.get('x_train')
y_train = data_dict.get('y_train')
x_test = data_dict.get('x_test')

depth = 2
lr = 0.017
n_est = 150

n_fold = 10

accuracy = np.zeros(n_fold)

base_learner = sk_tree.DecisionTreeClassifier(
    criterion='entropy', max_depth=depth)
model = AdaBoostClassifier(
    base_estimator=base_learner, n_estimators=n_est, learning_rate=lr)

cv = skl_ms.KFold(n_fold, shuffle=True, random_state=1)

for i, (train_index, val_index) in enumerate(cv.split(x_train)):
    x_train_l, x_val = x_train[train_index], x_train[val_index]
    y_train_l, y_val = y_train[train_index], y_train[val_index]

    model.fit(x_train_l, y_train_l)
    prediciton = model.predict(x_val)
    accuracy[i] = 1 - np.mean(prediciton != y_val)

avg_acc = np.mean(accuracy)
print('Accuracy:', accuracy)
print('Average: ', avg_acc, '\n')

prediction = predict(model, x_test)

if False:
    path = os.getcwd() + "/src/python/sml/mp/"
    txt_file = open(
        f'{path}manual_saved/{avg_acc:2f} | AdaBoost | lr={lr:2f} | d={depth} | n_est={n_est}.txt', 'w')
    txt_file.write(''.join(str(x) for x in prediction))
    txt_file.close()
    print(f'Write to file with acc: {avg_acc:2f}, d={depth}')
