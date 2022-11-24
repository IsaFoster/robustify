from _readData import getDataFromFile, getXandYFromFile, getXandYShortFromFile
from noiseCorruptions import doAll
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
import pandas as pd
import pickle

'************* Load Data *******************'
X_train, y_train, _, _ = getXandYFromFile()
X_train_short, y_train_short, _, _ = getXandYShortFromFile()
'*******************************************'

'********** Make and save models ***********'
modelName = "SVC_full_set"
model = SVC(kernel='linear')
model.fit(X_train, y_train.values.ravel())
pickle.dump(model, open('../Models/' + modelName, 'wb'))

modelName = "SVC_reduced_set"
model = SVC(kernel='linear')
model.fit(X_train_short, y_train_short.values.ravel())
pickle.dump(model, open('../Models/' + modelName, 'wb'))
'*******************************************'