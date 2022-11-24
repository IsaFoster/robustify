from _readData import getDataFramesFromFile, getXandYFromFile
from noiseCorruptions import noiseCorruptions
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
import pandas as pd
import pickle
import random

seed = 39
random.seed(seed)
np.random.seed(seed=seed)

df_train, _, _ = getDataFramesFromFile()
__, _, X_test, y_test = getXandYFromFile()

modelName = 'SVC_reduced_set'
model = pickle.load(open('../Models/' + modelName, 'rb'))
n_repeats = 10
random_state = 44

noiseCorruptions(df_train, X_test, y_test, model, corruptions=10)