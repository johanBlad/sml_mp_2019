import os
import numpy as np
import random as rnd

import sklearn.linear_model as skl_lm
import sklearn.model_selection as mod_sel
import sklearn.tree as sk_tree
from sklearn.ensemble import AdaBoostClassifier

from utility import initialize_data, predict

path = os.getcwd() + "/src/python/sml/mp/"
data_dict = initialize_data()

x_train = data_dict.get('x_train')
y_train = data_dict.get('y_train')
x_test = data_dict.get('x_test')


depths = [1, 2]
estimater_choices = range(100, 700, 50)

while True:
    lr = rnd.random()/rnd.randint(1,10)
    depth = rnd.choice(depths)
    n_est = rnd.choice(estimater_choices)

    base_learner = sk_tree.DecisionTreeClassifier(criterion='entropy', max_depth=depth)
    AdaBoost = AdaBoostClassifier(base_estimator=base_learner, n_estimators=n_est, learning_rate=lr)
    model = AdaBoost.fit(x_train, y_train)
    val_stats = mod_sel.cross_validate(model, x_train, y=y_train, cv=10)

    avg_score = np.mean(val_stats.get('test_score'))
    prediction = predict(model, x_test)
    if ((avg_score > 0.82 and depth == 1) or (avg_score > 0.83 and depth == 2)):
        txt_file = open(f'{path}saved_models/{avg_score:2f} | AdaBoost | lr={lr:2f} | d={depth} | n_est={n_est}.txt', 'w')
        txt_file.write(''.join(str(x) for x in prediction))
        txt_file.close()
        print(f'Write to file with acc: {avg_score:2f}, d={depth}')

print("DONE")

