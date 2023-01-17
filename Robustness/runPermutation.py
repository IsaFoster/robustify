from Setup._readData import getDataFramesFromFile, getXandYFromFile, getXandYShortFromFile
from permutationImportance import permutationImportance, meanAccuracyDecrease
import numpy as np
import random
import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer

seed = 39
random.seed(seed)
np.random.seed(seed=seed)

n_repeats = 100
random_state = 44

_, _, X_test, y_test = getXandYShortFromFile()

def accuracy_keras(estimator, X, y):
    pred = estimator.predict(X) 
    pred = np.argmax(pred, axis = 1)
    return accuracy_score(y, pred)

def accuacy_keras_2(y_true, y_pred):
    pred = np.argmax(y_pred, axis = 1)
    return accuracy_score(y_true, pred)

def permuteModel(modelName, X_test, y_test, n_repeats, random_state, scoring):
    model = pickle.load(open('../Models/' + modelName, 'rb'))

    permutationImportance(model, X_test, y_test, n_repeats, random_state, scoring)
    meanAccuracyDecrease(model, X_test, y_test, n_repeats, random_state, scoring)

#permuteModel('RF_full_set', X_test, y_test, n_repeats, random_state)
#permuteModel('SVC_full_set', X_test, y_test, n_repeats, random_state)
#permuteModel('LDA_full_set', X_test, y_test, n_repeats, random_state)
permuteModel('SK_reduced_set', X_test, y_test, n_repeats, random_state, scoring='r2')
#permuteModel('SK_reduced_set', X_test, y_test, n_repeats, random_state, scoring=make_scorer(accuacy_keras_2, greater_is_better=True))