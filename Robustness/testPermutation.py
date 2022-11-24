from _readData import getDataFramesFromFile, getXandYFromFile
from permutationImportance import permutationImportance, meanAccuracyDecrease
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import pandas as pd
import random
import pickle

seed = 39
random.seed(seed)
np.random.seed(seed=seed)

__, _, X_test, y_test = getXandYFromFile()
modelName = 'SVC_reduced_set'
model = pickle.load(open('../Models/' + modelName, 'rb'))
n_repeats = 10
random_state = 44

permutationImportance(model, X_test, y_test, n_repeats, random_state)
meanAccuracyDecrease(model, X_test, y_test, n_repeats, random_state)