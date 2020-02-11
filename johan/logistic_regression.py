from utility import initialize_data, predict
import sklearn.linear_model as skl_lm
import sklearn.model_selection as mod_sel
import numpy as np

data_dict = initialize_data()

x_train = data_dict.get('x_train')
y_train = data_dict.get('y_train')
x_test = data_dict.get('x_test')

model = skl_lm.LogisticRegression(solver='lbfgs').fit(x_train, y_train)
val_stats = mod_sel.cross_validate(model, x_train, y=y_train, cv=5)

print('\nTest scores:', val_stats.get('test_score'))
print('Average: ', np.mean(val_stats.get('test_score')), '\n')

prediction = predict(model, x_test)


