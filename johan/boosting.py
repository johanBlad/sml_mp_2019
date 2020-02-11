from utility import initialize_data, predict
import sklearn.linear_model as skl_lm
import sklearn.model_selection as mod_sel
import sklearn.tree as sk_tree
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
import os

data_dict = initialize_data()

x_train = data_dict.get('x_train')
y_train = data_dict.get('y_train')
x_test = data_dict.get('x_test')

depth = 1
lr = 0.035
n_est = 350

base_learner = sk_tree.DecisionTreeClassifier(
    criterion='entropy', max_depth=depth)
AdaBoost = AdaBoostClassifier(
    base_estimator=base_learner, n_estimators=n_est, learning_rate=lr)
model = AdaBoost.fit(x_train, y_train)

val_stats = mod_sel.cross_validate(model, x_train, y=y_train, cv=10)
avg_score = np.mean(val_stats.get('test_score'))

print('\nTest scores:', val_stats.get('test_score'))
print('Average: ', avg_score, '\n')

prediction = predict(model, x_test)

if True:
    path = os.getcwd() + "/src/python/sml/mp/"
    txt_file = open(
        f'{path}manual_saved/{avg_score:2f} | AdaBoost | lr={lr:2f} | d={depth} | n_est={n_est}.txt', 'w')
    txt_file.write(''.join(str(x) for x in prediction))
    txt_file.close()
    print(f'Write to file with acc: {avg_score:2f}, d={depth}')

print(prediction)
