from _readData import getDataFramesFromFile, getXandYFromFile, getXandYShortFromFile
from permutationImportance import permutationImportance, meanAccuracyDecrease
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import pandas as pd
import random
import pickle
from sklearn.metrics import accuracy_score

seed = 39
random.seed(seed)
np.random.seed(seed=seed)

n_repeats = 100
random_state = 44

_, _, X_test, y_test = getXandYShortFromFile()


def permuteModel(modelName, X_test, y_test, n_repeats, random_state):
    model = pickle.load(open('../Models/' + modelName, 'rb'))

    permutationImportance(model, X_test, y_test, n_repeats, random_state)
    meanAccuracyDecrease(model, X_test, y_test, n_repeats, random_state)

permuteModel('RF_full_set', X_test, y_test, n_repeats, random_state)
permuteModel('SVC_full_set', X_test, y_test, n_repeats, random_state)
permuteModel('LDA_full_set', X_test, y_test, n_repeats, random_state)