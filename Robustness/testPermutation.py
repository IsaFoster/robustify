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

_, _, X_test, y_test = getXandYShortFromFile()
modelName = 'RF_reduced_set'

#_, _, X_test, y_test = getXandYFromFile()
#modelName = 'SVC_full_set'

model = pickle.load(open('../Models/' + modelName, 'rb'))
n_repeats = 10
random_state = 44

permutationImportance(model, X_test, y_test, n_repeats, random_state)
meanAccuracyDecrease(model, X_test, y_test, n_repeats, random_state)