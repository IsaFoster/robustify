from Setup._readData import getDataFramesFromFile, getDataFramesShortFromFile, getXandYFromFile, getXandYShortFromFile
from noiseCorruptions import noiseCorruptions
import numpy as np
import pandas as pd
import pickle
import random


seed = 39
random.seed(seed)
np.random.seed(seed=seed)

df_train, _ = getDataFramesShortFromFile()
_, _, X_test, y_test = getXandYShortFromFile()


def corruptModel(modelName, df_train, X_test, y_test, random_state, corruptions):
    model = pickle.load(open('../Models/' + modelName, 'rb'))
    
    noiseCorruptions(df_train, X_test, y_test, model, random_state, corruptions)


corruptModel('RF_reduced_set', df_train, X_test, y_test, 50, 100)
corruptModel('SVC_reduced_set', df_train, X_test, y_test, 50, 100)
corruptModel('LDA_reduced_set', df_train, X_test, y_test, 50, 100)


#corruptModel('SVC_reduced_set', df_train, X_test, y_test, 50, 50)

# TODO: should not use pretrained model?