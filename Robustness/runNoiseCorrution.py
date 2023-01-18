from Setup._readData import getDataFramesFromFile, getDataFramesShortFromFile, getXandYFromFile, getXandYShortFromFile
from noiseCorruptions import noiseCorruptions
from noiseCorruptions_2 import corruptData, plotData, all
from noiseCorruptions_3 import noiseCorruptions_3
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
import pandas as pd
import pickle
import random
from sklearn.neural_network import MLPClassifier

seed = 39
random.seed(seed)
np.random.seed(seed=seed)

df_train, _ = getDataFramesShortFromFile()
_, _, X_test, y_test = getXandYShortFromFile()


'''def corruptModel(modelName, df_train, X_test, y_test, random_state, corruptions):
    model = pickle.load(open('../Models/' + modelName, 'rb'))
    
    noiseCorruptions(df_train, X_test, y_test, model, random_state, corruptions)


corruptModel('RF_full_set', df_train, X_test, y_test, 50, 100)
corruptModel('SVC_full_set', df_train, X_test, y_test, 50, 100)
corruptModel('LDA_full_set', df_train, X_test, y_test, 50, 100)

#corruptModel('SVC_reduced_set', df_train, X_test, y_test, 50, 50)

# TODO: should not use pretrained model?



my_dict = {
    'lep_1_pt': {'percentageShift': np.linspace(0, 10, 11)},  
    'lep_2_charge': {'flipSign'},
    'jet_n': {'addOrSubtract': [1, 2]}
}


def corruptModel(modelName, df_train, X_test, y_test, corruption_dict, random_state):
    model = pickle.load(open('../Models/' + modelName, 'rb'))
    
    noiseCorruptions_2(df_train, X_test, y_test, model, corruption_dict, random_state)


corruptModel('RF_reduced_set', df_train, X_test, y_test, my_dict, 50)
#corruptModel('SVC_reduced_set', df_train, X_test, y_test, my_dict, 50)
#corruptModel('LDA_reduced_set', df_train, X_test, y_test, my_dict, 50)
'''

my_dict = {
    'percentageShift': [['lep_1_pt', 'lep_2_eta', 'jet_1_phi'], np.linspace(0, 10, 11)],
    'flipSign': [['lep_2_charge', 'lep_1_charge']],
    'addOrSubtract': [['jet_n', 'alljet_n'], [1, 2]],
}

def corruptModel(modelName, df_train, X_test, y_test, corruption_dict, corruptions, random_state):
    model = pickle.load(open('../Models/' + modelName, 'rb'))
    
    all(df_train, X_test, y_test, model, corruption_dict, corruptions, random_state)

    #noiseCorruptions_3(df_train, X_test, y_test, model, corruption_dict, corruptions, random_state)


corruptModel('RF_reduced_set', df_train, X_test, y_test, my_dict, 10, 50)