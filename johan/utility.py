import os
import pandas as pd
import numpy as np
import sklearn.preprocessing as skl_pre


# Load data and return dataframes
def load_data():
    path = os.getcwd() + "/src/python/sml/mp/"
    test_data = pd.read_csv(path + "songs_to_classify.csv")
    train_data = pd.read_csv(path + "training_data.csv")
    return (train_data, test_data)


def initialize_data():
    train, test = load_data()

    x_train = train.iloc[:, :-1]
    y_train = train.iloc[:, -1]

    scaler = skl_pre.StandardScaler().fit(pd.concat([x_train, test]))
    return {'x_train': scaler.transform(x_train),
            'y_train': y_train,
            'x_test': scaler.transform(test),
            'train_raw': train,
            'test_raw': test
            }

def predict(model, x_test, threshold=0.5):
    predict_probabilities = model.predict_proba(x_test)
    prediction = np.empty(len(x_test), dtype=object)
    prediction = np.where(predict_probabilities[:, 0] >= threshold, 0, 1)
    return prediction

